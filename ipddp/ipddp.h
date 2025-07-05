#pragma once

#include "optimal_control_problem.h"
#include "param.h"
// #include <Eigen/Dense>
// #include <unsupported/Eigen/CXX11/Tensor>

#include <functional>
#include <cmath>
#include <ctime>

#include <iomanip>
#include <iostream>
#include <memory>
#include <type_traits>

#include <iostream>

template <typename Scalar>
class ALIPDDP {
public:
    explicit ALIPDDP(OptimalControlProblem<Scalar>& ocp_ref);
    ~ALIPDDP();

    void init(Param param);
    void solve();

    std::vector<Eigen::VectorXd> getResX();
    std::vector<Eigen::VectorXd> getResU();
    std::vector<double> getAllCost();

private:
    std::shared_ptr<OptimalControlProblem<Scalar>> ocp;

    // // Constraint Stack
    // std::vector<int> dim_hs_top; // Connic Constraint Head Stack
    // std::vector<int> dim_hTs_top; // Connic Constraint Head Stack (Terminal)
    // int dim_hs_max; // Maximum Dimension of Connic Constraint
    // int dim_hTs_max; // Maximum Dimension of Connic Constraint (Terminal)

    std::vector<Eigen::VectorXd> X; // State
    std::vector<Eigen::VectorXd> U; // Input
    
    std::vector<Eigen::VectorXd> Z; // Equality Lagrangian Multiplier
    std::vector<Eigen::VectorXd> R; // Equality Slack
    std::vector<Eigen::VectorXd> Y; // Inequality Lagrangian Multiplier
    std::vector<Eigen::VectorXd> S; // Inequality Slack
    std::vector<Eigen::VectorXd> C; // Inequality Constraint
    std::vector<Eigen::VectorXd> EC; // Equality Constraint

    Eigen::VectorXd ZT; // Equality Lagrangian Multiplier (Terminal)
    Eigen::VectorXd RT; // Equality Slack (Terminal)
    Eigen::VectorXd YT; // Inequality Lagrangian Multiplier (Terminal)
    Eigen::VectorXd ST; // Inequality Slack (Terminal)
    Eigen::VectorXd CT; // Inequality Constraint (Terminal)
    Eigen::VectorXd ECT; // Equality Constraint (Terminal)

    std::vector<bool> is_c_active;
    std::vector<bool> is_ec_active;
    bool is_c_active_all;
    bool is_ec_active_all;
    bool is_cT_active;
    bool is_ecT_active;

    Eigen::VectorXd scale_ec;
    Eigen::VectorXd scale_c;
    Eigen::VectorXd scale_ecT;
    Eigen::VectorXd scale_cT;

    std::vector<Eigen::VectorXd> lambda;
    Eigen::VectorXd lambdaT;
    std::vector<int> dim_rn;
    int dim_rnT;
    Eigen::VectorXd perturb(const Eigen::VectorXd& xn, const Eigen::VectorXd& x);
    
    double cost;
    Param param;
    void initialRoll();
    void resetFilter();
    double logcost;
    double error;
    
    std::vector<double> step_list; // Step Size List
    int step; // Step Size Index
    int forward_failed;
    bool is_diff_calculated;

    int update_counter;
    int iter;
    int inner_iter;
    void resetRegulation();
    int regulate;
    bool backward_failed;

    std::vector<Eigen::MatrixXd> fx_all;
    std::vector<Eigen::MatrixXd> fu_all;
    // std::vector<std::vector<Eigen::MatrixXd>> fxx_all;
    // std::vector<std::vector<Eigen::MatrixXd>> fxu_all;
    // std::vector<std::vector<Eigen::MatrixXd>> fuu_all;
    Eigen::VectorXd px_all;
    Eigen::MatrixXd pxx_all;
    std::vector<Eigen::VectorXd> qx_all;
    std::vector<Eigen::VectorXd> qu_all;
    std::vector<Eigen::MatrixXd> qxx_all;
    std::vector<Eigen::MatrixXd> qxu_all;
    std::vector<Eigen::MatrixXd> quu_all;
    std::vector<Eigen::MatrixXd> cx_all;
    std::vector<Eigen::MatrixXd> cu_all;
    std::vector<Eigen::MatrixXd> ecx_all;
    std::vector<Eigen::MatrixXd> ecu_all;
    Eigen::MatrixXd cTx_all;
    Eigen::MatrixXd ecTx_all;

    std::vector<Eigen::VectorXd> ku; // Input Feedforward Gain 
    std::vector<Eigen::VectorXd> kr; // Equality Slack Feedforward Gain
    std::vector<Eigen::VectorXd> kz; // Equality Lagrangian Multiplier Feedforward Gain
    std::vector<Eigen::VectorXd> ks; // Inequality Slack Feedforward Gain
    std::vector<Eigen::VectorXd> ky; // Inequality Lagrangian Multiplier Feedforward Gain
    
    std::vector<Eigen::MatrixXd> Ku; // Input Feedback Gain
    std::vector<Eigen::MatrixXd> Kr; // Equality Slack Feedback Gain
    std::vector<Eigen::MatrixXd> Kz; // Equality Lagrangian Multiplier Feedback Gain
    std::vector<Eigen::MatrixXd> Ks; // Inequality Slack Feedback Gain
    std::vector<Eigen::MatrixXd> Ky; // Inequality Lagrangian Multiplier Feedback Gain

    Eigen::VectorXd krT; // Equality Slack Feedforward Gain (Terminal)
    Eigen::VectorXd kzT; // Equality Lagrangian Multiplier Feedforward Gain (Terminal)
    Eigen::MatrixXd KrT; // Equality Slack Feedback Gain (Terminal)
    Eigen::MatrixXd KzT; // Equality Lagrangian Multiplier Feedback Gain (Terminal)

    Eigen::VectorXd ksT; // Inequality Slack Feedforward Gain (Terminal)
    Eigen::VectorXd kyT; // Inequality Lagrangian Multiplier Feedforward Gain (Terminal)
    Eigen::MatrixXd KsT; // Inequality Slack Feedback Gain (Terminal)
    Eigen::MatrixXd KyT; // Inequality Lagrangian Multiplier Feedback Gain (Terminal)

    std::vector<Eigen::VectorXd> e;
    Eigen::VectorXd eT;

    double opterror;
    double opterror_rpT_ec;
    double opterror_rdT_ec;
    double opterror_rpT_c;
    double opterror_rdT_c;
    double opterror_rp_c;
    double opterror_rd_c;
    Eigen::VectorXd dV; // Expected Value Change

    std::vector<double> all_cost;

    // Algorithm
    Eigen::MatrixXd L(const Eigen::VectorXd& x);
    void calculateAllDiff();
    void backwardPass();
    void checkRegulate();
    void L_inv_times_vec(Eigen::Ref<Eigen::VectorXd> out, const Eigen::Ref<const Eigen::VectorXd>& soc, const Eigen::Ref<const Eigen::VectorXd>& vec);
    void L_times_vec(Eigen::Ref<Eigen::VectorXd> out, const Eigen::Ref<const Eigen::VectorXd>& soc, const Eigen::Ref<const Eigen::VectorXd>& vec);
    void L_inv(Eigen::Ref<Eigen::MatrixXd> inv, const Eigen::Ref<const Eigen::VectorXd>& soc);
    void L_inv_arrow(Eigen::Ref<Eigen::MatrixXd> out, const Eigen::Ref<const Eigen::MatrixXd>& L_inv, const Eigen::Ref<const Eigen::VectorXd>& soc);
    void L_inv_times_arrow(Eigen::Ref<Eigen::MatrixXd> out, const Eigen::Ref<const Eigen::VectorXd>& soc_inv, const Eigen::Ref<const Eigen::VectorXd>& soc_arrow);
    void forwardPass();
    double calculateTotalCost(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U);
    void logPrint();
};

template <typename Scalar>
ALIPDDP<Scalar>::ALIPDDP(OptimalControlProblem<Scalar>& ocp_ref) : ocp(std::make_shared<OptimalControlProblem<Scalar>>(ocp_ref)) {
    is_c_active.resize(ocp->N);
    is_ec_active.resize(ocp->N);
    is_c_active_all = false;
    is_ec_active_all = false;
    for (int k = 0; k < ocp->N; ++k) {
        is_c_active[k] = (ocp->dim_c[k][0] + ocp->dim_c[k][1] > 0);
        is_ec_active[k] = (ocp->dim_ec[k] > 0);
        is_c_active_all = (is_c_active_all || is_c_active[k]);
        is_ec_active_all = (is_ec_active_all || is_ec_active[k]);
    }
    is_cT_active = ((ocp->dim_cT[0] + ocp->dim_cT[1]) > 0);
    is_ecT_active = (ocp->dim_ecT > 0);

    // struct ConstraintActiveFlag {
    //     std::vector<bool> stage_c;
    //     std::vector<bool> stage_ec;
    //     bool stage_c_any;
    //     bool stage_ec_any;
    //     bool term_c;
    //     bool term_ec;
    // };

    X.resize(ocp->N + 1);
    U.resize(ocp->N);
    Z.resize(ocp->N);
    R.resize(ocp->N);
    Y.resize(ocp->N);
    S.resize(ocp->N);
    C.resize(ocp->N);
    EC.resize(ocp->N);
    e.resize(ocp->N);
    for(int k = 0; k < ocp->N; ++k) {
        if (ocp->is_x_initialized[k]) {X[k] = ocp->X0[k];}
        else {X[k].setZero(ocp->dim_x[k]);}
        if (ocp->is_u_initialized[k]) {U[k] = ocp->U0[k];}
        else {U[k].setZero(ocp->dim_u[k]);}
        R[k].setZero(ocp->dim_ec[k]);
        Z[k].setZero(ocp->dim_ec[k]);
        S[k].setZero(ocp->dim_c[k][0] + ocp->dim_c[k][1]);
        Y[k].setZero(ocp->dim_c[k][0] + ocp->dim_c[k][1]);
        e[k].setZero(ocp->dim_c[k][0] + ocp->dim_c[k][1]);
        if (ocp->dim_c[k][0]) {
            S[k].head(ocp->dim_c[k][0]) = Eigen::VectorXd::Ones(ocp->dim_c[k][0]);
            Y[k].head(ocp->dim_c[k][0]) = Eigen::VectorXd::Ones(ocp->dim_c[k][0]);
            e[k].head(ocp->dim_c[k][0]) = Eigen::VectorXd::Ones(ocp->dim_c[k][0]);
        }
        for (auto dim_h_top : ocp->dim_hs_top[k]) {
            S[k](dim_h_top) = 1.0;
            Y[k](dim_h_top) = 1.0;
            e[k](dim_h_top) = 1.0;
        }
    }
    if (ocp->is_x_initialized[ocp->N]) {X[ocp->N] = ocp->X0[ocp->N];}

    RT.setZero(ocp->dim_ecT);
    ZT.setZero(ocp->dim_ecT);
    ST.setZero(ocp->dim_cT[0] + ocp->dim_cT[1]);
    YT.setZero(ocp->dim_cT[0] + ocp->dim_cT[1]);
    eT.setZero(ocp->dim_cT[0] + ocp->dim_cT[1]);
    if (ocp->dim_cT[0]) {
        ST.head(ocp->dim_cT[0]) = Eigen::VectorXd::Ones(ocp->dim_cT[0]);
        YT.head(ocp->dim_cT[0]) = Eigen::VectorXd::Ones(ocp->dim_cT[0]);
        eT.head(ocp->dim_cT[0]) = Eigen::VectorXd::Ones(ocp->dim_cT[0]);
    }
    for (auto dim_h_top : ocp->dim_hTs_top) {
        ST(dim_h_top) = 1.0;
        YT(dim_h_top) = 1.0;
        eT(dim_h_top) = 1.0;
    }

    fx_all.resize(ocp->N);
    fu_all.resize(ocp->N);
    qx_all.resize(ocp->N);
    qu_all.resize(ocp->N);
    qxx_all.resize(ocp->N);
    qxu_all.resize(ocp->N);
    quu_all.resize(ocp->N);
    // fxx_all.resize(ocp->N);
    // fxu_all.resize(ocp->N);
    // fuu_all.resize(ocp->N);
    cx_all.resize(ocp->N);
    cu_all.resize(ocp->N);
    ecx_all.resize(ocp->N);
    ecu_all.resize(ocp->N);

    ku.resize(ocp->N);
    kr.resize(ocp->N);
    kz.resize(ocp->N);
    ks.resize(ocp->N);
    ky.resize(ocp->N);
    Ku.resize(ocp->N);
    Kr.resize(ocp->N);
    Kz.resize(ocp->N);
    Ks.resize(ocp->N);
    Ky.resize(ocp->N);
}

template <typename Scalar>
ALIPDDP<Scalar>::~ALIPDDP() {
}

template <typename Scalar>
void ALIPDDP<Scalar>::init(Param param) {
    this->param = param;

    // Only support same dynamics
    dim_rn.resize(ocp->N);
    if (this->param.is_quaternion_in_state) {
        if (!this->param.quaternion_index) {
            throw std::runtime_error("ALIPDDP: quaternion_index must be initialized. (see is_quaternion_in_state)");
        }
        else {
            for(int k = 0; k < ocp->N; ++k) {
                dim_rn[k] = ocp->dim_x[k] - 1;
            }
            dim_rnT = ocp->dim_xT - 1;
        }
    }
    else {
        for(int k = 0; k < ocp->N; ++k) {
            dim_rn[k] = ocp->dim_x[k];
        }
        dim_rnT = ocp->dim_xT;
    }

    for (int k = 0; k < ocp->N; ++k) {
        ku[k] = Eigen::VectorXd::Zero(ocp->dim_u[k]);
        Ku[k] = Eigen::MatrixXd::Zero(ocp->dim_u[k], dim_rn[k]);
        if (is_c_active[k]) {
            ks[k] = Eigen::VectorXd::Zero(ocp->dim_c[k][0] + ocp->dim_c[k][1]);
            ky[k] = Eigen::VectorXd::Zero(ocp->dim_c[k][0] + ocp->dim_c[k][1]);
            Ks[k] = Eigen::MatrixXd::Zero(ocp->dim_c[k][0] + ocp->dim_c[k][1], dim_rn[k]);
            Ky[k] = Eigen::MatrixXd::Zero(ocp->dim_c[k][0] + ocp->dim_c[k][1], dim_rn[k]);
        }
        if (is_ec_active[k]) {
            kr[k] = Eigen::VectorXd::Zero(ocp->dim_ec[k]);
            kz[k] = Eigen::VectorXd::Zero(ocp->dim_ec[k]);
            Kr[k] = Eigen::MatrixXd::Zero(ocp->dim_ec[k], dim_rn[k]);
            Kz[k] = Eigen::MatrixXd::Zero(ocp->dim_ec[k], dim_rn[k]);
        }
    }

    initialRoll();
    
    // check dim_c
    // if (this->param.mu == 0) {this->param.mu = cost / ocp->N / dim_c;} // Auto Select

    lambda.resize(ocp->N);
    for(int k = 0; k < ocp->N; ++k) {
        if (ocp->dim_ec[k]) {lambda[k].setZero(ocp->dim_ec[k]);}
    }
    lambdaT.setZero(ocp->dim_ecT);

    resetFilter();
    resetRegulation();

    for (int i = 0; i <= this->param.max_step_iter; ++i) {
        step_list.push_back(std::pow(2.0, static_cast<double>(-i)));
    }
}

template <typename Scalar>
void ALIPDDP<Scalar>::initialRoll() {
    for (int k = 0; k < ocp->N; ++k) {
        const Eigen::VectorXd& xk = X[k];
        const Eigen::VectorXd& uk = U[k];
        if (ocp->dim_c[k][0] || ocp->dim_c[k][1]) {C[k] = ocp->c[k](xk, uk);}
        if (ocp->dim_ec[k]) {EC[k] = ocp->ec[k](xk, uk);}
        X[k + 1] = ocp->dynamics_seq[k]->f(xk, uk);
    }
    if (ocp->dim_cT[0] || ocp->dim_cT[1]) {CT = ocp->cT(X[ocp->N]);}
    if (ocp->dim_ecT) {ECT = ocp->ecT(X[ocp->N]);}
    
    cost = calculateTotalCost(X, U);
}

template <typename Scalar>
void ALIPDDP<Scalar>::resetFilter() {
    double barriercost = 0.0;
    double barriercostT = 0.0;
    double alcost = 0.0;
    double alcostT = 0.0;
    
    // CHECK: Possible to use GEMM if we use same dynamics & constraints across all timestep
    // Implement 2 way method?
    
    for (int k = 0; k < ocp->N; ++k) {
        if (ocp->dim_c[k][0]) {
            barriercost += S[k].head(ocp->dim_c[k][0]).array().log().sum();
        }
        for (int i = 0; i < ocp->dim_hs[k].size(); ++i) {
            const int d = ocp->dim_hs[k][i];
            const int idx = ocp->dim_hs_top[k][i];
            const double s = S[k](idx);
            const double v = S[k].segment(idx + 1, d - 1).squaredNorm();
            barriercost += 0.5 * log(s*s - v);
        }
    }
    if (ocp->dim_cT[0]) {
        barriercostT += ST.head(ocp->dim_cT[0]).array().log().sum();
    }
    for (int i = 0; i < ocp->dim_hTs.size(); ++i) {
        const int d = ocp->dim_hTs[i];
        const int idx = ocp->dim_hTs_top[i];
        const double s = ST(idx);
        const double v = ST.segment(idx + 1, d - 1).squaredNorm();
        barriercostT += 0.5 * log(s*s - v);
    }
    for (int k = 0; k < ocp->N; ++k) {
        if (ocp->dim_ec[k]) {
            alcost += (lambda[k].transpose() * R[k]).sum() + (0.5 * param.rho * R[k].squaredNorm());
        }
    }
    if (ocp->dim_ecT) {
        alcostT += (lambdaT.transpose() * RT) + (0.5 * param.rho * RT.squaredNorm());
    }
    logcost = cost - (param.mu * barriercost + param.muT * barriercostT) + (alcost + alcostT);

    error = 0.0;
    for (int k = 0; k < ocp->N; ++k) {
        if (ocp->dim_ec[k]) { error += (EC[k] + R[k]).array().abs().sum(); }
        if (ocp->dim_c[k][0] || ocp->dim_c[k][1]) { error += (C[k] + S[k]).array().abs().sum(); }
    }
    if (ocp->dim_ecT) { error += (ECT + RT).array().abs().sum(); }
    if (ocp->dim_cT[0] || ocp->dim_cT[1]) { error += (CT + ST).array().abs().sum(); }
    error = std::max(param.tolerance, error);
    
    step = 0;
    forward_failed = false;
}

template <typename Scalar>
void ALIPDDP<Scalar>::resetRegulation() {
    this->regulate = 0;
    this->backward_failed = false;
}

template <typename Scalar>
double ALIPDDP<Scalar>::calculateTotalCost(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U) {
    double cost = 0.0;
    for (int k = 0; k < ocp->N; ++k) {
        cost += ocp->cost_seq[k]->q(X[k], U[k]);
    }
    cost += ocp->terminal_cost->p(X[ocp->N]);
    return cost;
}

template <typename Scalar>
void ALIPDDP<Scalar>::solve() {
    update_counter = 0;
    iter = 0;
    is_diff_calculated = false;
    
    clock_t start;
    clock_t finish;
    double duration;

    std::cout << std::setw(4) << "o_it"
    << std::setw(4) << "i_it"
    << std::setw(3) << "bp"
    << std::setw(3) << "fp"
    << std::setw(7) << "mu"
    << std::setw(16) << "rho"
    << std::setw(25) << "LogCost"
    << std::setw(22) << "OptError"
    << std::setw(18) << "Error"
    << std::setw(4) << "Reg"
    << std::setw(7) << "Step"
    << std::setw(5) << "Upt" << std::endl;

    // Outer Loop (Augmented Lagrangian & Iterior Point Method)
    while (true) {
        inner_iter = 0;
        // Inner Loop (Differential Dynamic Programming)
        while (inner_iter < this->param.max_inner_iter) {
            if (param.max_iter < ++iter) {break;}

            if (!is_diff_calculated) {
                this->calculateAllDiff();
                is_diff_calculated = true;
            }

            this->backwardPass();

            if (backward_failed) {
                this->logPrint();
                if (regulate==param.max_regularization) {break;}
                else {continue;}
            }
            
            this->forwardPass();
            if (!forward_failed) {
                is_diff_calculated = false;
                inner_iter++;
                update_counter++;
            }
            
            this->logPrint();
            
            all_cost.push_back(cost);
    
            // CHECK
            if (std::max(opterror, param.mu) <= param.tolerance) {
                // if (opterror <= param.tolerance) {
                std::cout << "Optimal Solution" << std::endl;
                return;
            }
    
            if (forward_failed && regulate==param.max_regularization) {
                break;
            }
    
            if ((opterror <= std::max(10.0 * param.mu, param.tolerance))) {
                break;
            }

            {
                // Seperate time frame?
                // If then, consider active marker to be vector (like switch condition)
                bool updated = false;
                if (is_c_active_all && opterror_rp_c < param.tolerance && opterror_rd_c < param.tolerance) {
                    if (param.mu > param.mu_min) {updated = true;}
                    param.mu = std::max(param.mu_min, std::min(param.mu_mul * param.mu, std::pow(param.mu, param.mu_exp)));
                }
                if (is_cT_active && opterror_rpT_c < param.tolerance && opterror_rdT_c < param.tolerance) {
                    if (param.muT > param.mu_min) {updated = true;}
                    param.muT = std::max(param.mu_min, std::min(param.mu_mul * param.muT, std::pow(param.muT, param.mu_exp)));
                }
                if (is_ecT_active && opterror_rpT_ec < param.tolerance && opterror_rdT_ec < param.tolerance) {
                    if (param.rho < param.rho_max) {updated = true;}
                    param.rho = std::min(param.rho_max, param.rho_mul * param.rho);
                    lambdaT = lambdaT + param.rho * RT;
                }
                if (updated) {
                    resetFilter();
                    inner_iter = 0;
                }
                // std::cout << "opterror_rp_c = " << opterror_rp_c << std::endl;
                // std::cout << "opterror_rd_c = " << opterror_rd_c << std::endl;
                // std::cout << "opterror_rpT_c = " << opterror_rpT_c << std::endl;
                // std::cout << "opterror_rdT_c = " << opterror_rdT_c << std::endl;
                // std::cout << "opterror_rpT_ec = " << opterror_rpT_ec << std::endl;
                // std::cout << "opterror_rdT_ec = " << opterror_rdT_ec << std::endl;
            }
        }
        if (param.max_iter < iter) {
            std::cout << "Max Iteration" << std::endl;
            return;
        }

        bool mu_stop = (param.mu <= param.mu_min);
        bool muT_stop = (param.muT <= param.mu_min);
        bool rho_stop = (param.rho >= param.rho_max);

        bool c_done   = (!is_c_active_all || mu_stop);
        bool ec_done   = (!is_ec_active_all || rho_stop);
        bool cT_done  = (!is_cT_active || muT_stop);
        bool ecT_done = (!is_ecT_active || rho_stop);
      
        if (c_done && cT_done && ec_done && ecT_done) {
            std::cout << "Outer Max/Min" << std::endl;
            return;
        }

        // Update Outer Loop Parameters
        if (is_c_active_all) {param.mu = std::max(param.mu_min, std::min(param.mu_mul * param.mu, std::pow(param.mu, param.mu_exp)));}
        if (is_ec_active_all || is_ecT_active) {param.rho = std::min(param.rho_max, param.rho_mul * param.rho);}
        if (is_cT_active) {param.muT = std::max(param.mu_min, std::min(param.mu_mul * param.muT, std::pow(param.muT, param.mu_exp)));}
        if (is_ecT_active) {lambdaT = lambdaT + param.rho * RT;}
        resetFilter();
        resetRegulation();
    }
}

template <typename Scalar>
void ALIPDDP<Scalar>::calculateAllDiff() {
    // CHECK: Multithreading (TODO: with CUDA)
    // CHECK 1: Making branch in for loop is fine for parallelization?
    // CHECK 2: Move to Model to make solver only consider Eigen (not autodiff::dual)

    // #pragma omp parallel for
    for (int k = 0; k < ocp->N; ++k) {
        Eigen::VectorXd x = X[k];
        Eigen::VectorXd u = U[k];

        fx_all[k] = ocp->dynamics_seq[k]->fx(x, u);
        fu_all[k] = ocp->dynamics_seq[k]->fu(x, u);
        qx_all[k] = ocp->cost_seq[k]->qx(x, u);
        qu_all[k] = ocp->cost_seq[k]->qu(x, u);
        qxx_all[k] = ocp->cost_seq[k]->qxx(x, u);
        quu_all[k] = ocp->cost_seq[k]->quu(x, u);
        qxu_all[k] = ocp->cost_seq[k]->qxu(x, u);
        if (is_c_active[k]) {
            cx_all[k] = ocp->cx[k](x, u);
            cu_all[k] = ocp->cu[k](x, u);
        }
        if (is_ec_active[k]) {
            ecx_all[k] = ocp->ecx[k](x, u);
            ecu_all[k] = ocp->ecu[k](x, u);
        }
    }
    Eigen::VectorXd xT = X[ocp->N];

    px_all = ocp->terminal_cost->px(xT);
    pxx_all = ocp->terminal_cost->pxx(xT);

    if (ocp->dim_cT[0] + ocp->dim_cT[1] > 0) {
        cTx_all = ocp->cTx(xT);
    }
    if (ocp->dim_ecT > 0) {
        ecTx_all = ocp->ecTx(xT);
    }
}

template <typename Scalar>
void ALIPDDP<Scalar>::backwardPass() {
    Eigen::VectorXd Vx;
    Eigen::MatrixXd Vxx;

    Eigen::VectorXd Qx, Qu;
    Eigen::MatrixXd Qxx, Qxu, Quu;
    Eigen::MatrixXd hat_Quu;

    Eigen::VectorXd rp, rd;
    
    Eigen::VectorXd Sinv_r;
    Eigen::MatrixXd Sinv_Y_Qyx, Sinv_Y_Qyu;
    
    Eigen::LLT<Eigen::MatrixXd> Quu_llt;

    opterror = 0.0;
    opterror_rpT_ec = 0.0;
    opterror_rdT_ec = 0.0;
    opterror_rpT_c = 0.0;
    opterror_rdT_c = 0.0;
    opterror_rp_c = 0.0;
    opterror_rd_c = 0.0;

    dV = Eigen::VectorXd::Zero(2);

    checkRegulate();

    double reg1_mu = param.reg1_min * (std::pow(param.reg1_exp, regulate));
    double reg2_mu = param.reg2_min * (std::pow(param.reg2_exp, regulate));

    Vx = px_all;
    Vxx = pxx_all;

    // Inequality Terminal Constraint
    if (is_cT_active) {
        const int dim_cT = ocp->dim_cT[0] + ocp->dim_cT[1];
        const int dim_gT = ocp->dim_cT[0];

        Eigen::Ref<const Eigen::MatrixXd> QyxT = cTx_all;

        Eigen::VectorXd rpT = CT + ST;
        Eigen::VectorXd rdT(dim_cT);
        rdT.head(dim_gT) = ST.head(dim_gT).cwiseProduct(YT.head(dim_gT));
        for (int i = 0; i < ocp->dim_hTs.size(); ++i) {
            const int d = ocp->dim_hTs[i];
            const int idx = ocp->dim_hTs_top[i];
            L_times_vec(rdT.segment(idx, d), YT.segment(idx, d), ST.segment(idx, d));
        }
        rdT -= param.muT * eT;
        Eigen::VectorXd rT(dim_cT);
        rT.head(dim_gT) = YT.head(dim_gT).cwiseProduct(rpT.head(dim_gT));
        for (int i = 0; i < ocp->dim_hTs.size(); ++i) {
            const int d = ocp->dim_hTs[i];
            const int idx = ocp->dim_hTs_top[i];
            L_times_vec(rT.segment(idx, d), YT.segment(idx, d), rpT.segment(idx, d));
        }
        rT -= rdT;

        Eigen::VectorXd Sinv_rT(dim_cT);
        Sinv_rT.head(dim_gT) = rT.head(dim_gT).cwiseQuotient(ST.head(dim_gT));
        for (int i = 0; i < ocp->dim_hTs.size(); ++i) {
            const int d = ocp->dim_hTs[i];
            const int idx = ocp->dim_hTs_top[i];
            L_inv_times_vec(Sinv_rT.segment(idx, d), ST.segment(idx, d), rT.segment(idx, d));
        }
        
        Eigen::MatrixXd Sinv_Y_QyxT(dim_cT, dim_rnT);
        Sinv_Y_QyxT.topRows(dim_gT) = QyxT.topRows(dim_gT).array().colwise() * (YT.head(dim_gT).array() / ST.head(dim_gT).array());
        Eigen::MatrixXd Sinv_Y_hT_max(ocp->dim_hTs_max, ocp->dim_hTs_max);
        for (int i = 0; i < ocp->dim_hTs.size(); ++i) {
            const int d = ocp->dim_hTs[i];
            const int idx = ocp->dim_hTs_top[i];
            Eigen::Ref<Eigen::MatrixXd> Sinv_Y_hT = Sinv_Y_hT_max.topLeftCorner(d, d);
            L_inv_times_arrow(Sinv_Y_hT, ST.segment(idx, d), YT.segment(idx, d));
            Sinv_Y_QyxT.middleRows(idx, d) = Sinv_Y_hT * QyxT.middleRows(idx, d);
        }

        ksT = - rpT;
        KsT = - QyxT;
        
        kyT = Sinv_rT;
        KyT = Sinv_Y_QyxT;

        Vx += KyT.transpose() * CT + QyxT.transpose() * kyT;
        Vxx += QyxT.transpose() * KyT + KyT.transpose() * QyxT;

        // with slack
        // Eigen::VectorXd QsT = YTinv * rdT;
        // Eigen::VectorXd QyT = rpT;
        // Eigen::MatrixXd I_cT = Eigen::VectorXd::Ones(dim_cT).asDiagonal();
        // dV(0) += QsT.transpose() * ksT;
        // dV(0) += QyT.transpose() * kyT;
        // dV(1) += kyT.transpose() * I_cT * ksT; 

        // Vx += KyT.transpose() * QyT + QyxT.transpose() * kyT;
        // Vx += KsT.transpose() * QsT + KyT.transpose() * I_cT * ksT + KsT.transpose() * I_cT * kyT;

        // Vxx += QyxT.transpose() * KyT + KyT.transpose() * QyxT;
        // Vxx += KyT.transpose() * I_cT * KsT + KsT.transpose() * I_cT * KyT;

        opterror = std::max({rpT.lpNorm<Eigen::Infinity>(), rdT.lpNorm<Eigen::Infinity>(), opterror});
        opterror_rpT_c = std::max({rpT.lpNorm<Eigen::Infinity>(), opterror_rpT_c});
        opterror_rdT_c = std::max({rdT.lpNorm<Eigen::Infinity>(), opterror_rdT_c});
    }

    // Equality Terminal Constraint
    if (is_ecT_active) {
        // const int dim_ecT = ocp->dim_ecT;

        Eigen::Ref<const Eigen::MatrixXd> QzxT = ecTx_all;

        Eigen::VectorXd rpT = ECT + RT;
        Eigen::VectorXd rdT = ZT + lambdaT + (param.rho * RT);
        
        krT = - rpT;
        KrT = - QzxT;
        kzT = param.rho * rpT - rdT;
        KzT = param.rho * QzxT;

        // CHECK: New Value Decrement
        // dV(0) += kzT.transpose() * ECT;

        Vx += KzT.transpose() * ECT + QzxT.transpose() * kzT;
        Vxx += QzxT.transpose() * KzT + KzT.transpose() * QzxT;

        // with slack
        // Eigen::VectorXd QrT = rdT;
        // Eigen::VectorXd QzT = rpT;
        // Eigen::MatrixXd I_ecT = Eigen::VectorXd::Ones(dim_ecT).asDiagonal();
        // dV(0) += QrT.transpose() * krT;
        // dV(0) += QzT.transpose() * kzT;
        // dV(1) += kzT.transpose() * I_ecT * krT; // Qrr = 0
        
        // Vx += KzT.transpose() * QzT + QzxT.transpose() * kzT;
        // Vx += KrT.transpose() * QrT + KzT.transpose() * I_ecT * krT + KrT.transpose() * I_ecT * kzT;
        
        // Vxx += QzxT.transpose() * KzT + KzT.transpose() * QzxT;
        // Vxx += KzT.transpose() * I_ecT * KrT + KrT.transpose() * I_ecT * KzT;
        
        opterror = std::max({rpT.lpNorm<Eigen::Infinity>(), rdT.lpNorm<Eigen::Infinity>(), opterror});
        opterror_rpT_ec = std::max({rpT.lpNorm<Eigen::Infinity>(), opterror_rpT_ec});
        opterror_rdT_ec = std::max({rdT.lpNorm<Eigen::Infinity>(), opterror_rdT_ec});
    }

    backward_failed = false;

    for (int k = ocp->N - 1; k >= 0; --k) {
        Eigen::Ref<const Eigen::MatrixXd> fx = fx_all[k];
        Eigen::Ref<const Eigen::MatrixXd> fu = fu_all[k];

        Eigen::Ref<const Eigen::VectorXd> qx = qx_all[k];
        Eigen::Ref<const Eigen::VectorXd> qu = qu_all[k];

        Eigen::Ref<const Eigen::MatrixXd> qxx = qxx_all[k];
        Eigen::Ref<const Eigen::MatrixXd> qxu = qxu_all[k];
        Eigen::Ref<const Eigen::MatrixXd> quu = quu_all[k];

        Qx = qx + (fx.transpose() * Vx);
        Qu = qu + (fu.transpose() * Vx);
        
        // DDP (TODO: Vector-Hessian Product)
        // Qxx = qxx + (fx.transpose() * Vxx * fx) + tensdot(Vx, fxx);
        // Qxu = qxu + (fx.transpose() * Vxx * fu) + tensdot(Vx, fxu);
        // Quu = quu + (fu.transpose() * Vxx * fu) + tensdot(Vx, fuu);
        
        // iLQR
        // Qxx = qxx + (fx.transpose() * Vxx * fx);
        // Qxu = qxu + (fx.transpose() * Vxx * fu);
        // Quu = quu + (fu.transpose() * Vxx * fu);

        // Regularization
        Vxx.diagonal().array() += reg1_mu;
        Qxx = qxx + (fx.transpose() * Vxx * fx);
        Qxu = qxu + (fx.transpose() * Vxx * fu);
        Quu = quu + (fu.transpose() * Vxx * fu);
        Quu.diagonal().array() += reg2_mu;

        Eigen::Ref<Eigen::VectorXd> ku_ = ku[k];
        Eigen::Ref<Eigen::MatrixXd> Ku_ = Ku[k];

        ku_ = - Qu; // hat_Qu
        Ku_ = - Qxu.transpose(); // hat_Qxu
        hat_Quu = Quu;

        if (is_c_active[k]) {
            Eigen::Ref<Eigen::VectorXd> s = S[k];
            Eigen::Ref<Eigen::VectorXd> y = Y[k];
            Eigen::Ref<Eigen::VectorXd> c_v = C[k];
            
            Eigen::Ref<Eigen::MatrixXd> Qyx = cx_all[k];
            Eigen::Ref<Eigen::MatrixXd> Qyu = cu_all[k];
            
            const int dim_c = ocp->dim_c[k][0] + ocp->dim_c[k][1];
            const int dim_g = ocp->dim_c[k][0];

            Qx += Qyx.transpose() * y;
            Qu += Qyu.transpose() * y;
            
            rp = c_v + s;

            rd.resize(dim_c);
            rd.head(dim_g) = s.head(dim_g).cwiseProduct(y.head(dim_g));
            for (int i = 0; i < ocp->dim_hs[k].size(); ++i) {
                const int idx = ocp->dim_hs_top[k][i];
                const int n = ocp->dim_hs[k][i];
                L_times_vec(rd.segment(idx, n), y.segment(idx, n), s.segment(idx, n));
            }
            rd -= param.mu * e[k];

            Eigen::VectorXd r(dim_c);
            r.head(dim_g) = y.head(dim_g).cwiseProduct(rp.head(dim_g));
            for (int i = 0; i < ocp->dim_hs[k].size(); ++i) {
                const int d = ocp->dim_hs[k][i];
                const int idx = ocp->dim_hs_top[k][i];
                L_times_vec(r.segment(idx, d), y.segment(idx, d), rp.segment(idx, d));
            }
            r -= rd;

            Sinv_r.resize(dim_c);
            Sinv_r.head(dim_g) = r.head(dim_g).cwiseQuotient(s.head(dim_g));
            for (int i = 0; i < ocp->dim_hs[k].size(); ++i) {
                const int d = ocp->dim_hs[k][i];
                const int idx = ocp->dim_hs_top[k][i];
                L_inv_times_vec(Sinv_r.segment(idx, d), s.segment(idx, d), r.segment(idx, d));
            }

            // more complex, but more fast
            Sinv_Y_Qyx.resize(dim_c, dim_rn[k]);
            Sinv_Y_Qyu.resize(dim_c, ocp->dim_u[k]);
            Eigen::VectorXd Sinv_Y_g = y.head(dim_g).cwiseQuotient(s.head(dim_g));
            Sinv_Y_Qyx.topRows(dim_g) = Qyx.topRows(dim_g).array().colwise() * Sinv_Y_g.array();
            Sinv_Y_Qyu.topRows(dim_g) = Qyu.topRows(dim_g).array().colwise() * Sinv_Y_g.array();
            Eigen::MatrixXd SYinv_h_max(ocp->dim_hs_max[k], ocp->dim_hs_max[k]);
            for (int i = 0; i < ocp->dim_hs[k].size(); ++i) {
                const int d = ocp->dim_hs[k][i];
                const int idx = ocp->dim_hs_top[k][i];
                Eigen::Ref<Eigen::MatrixXd> Sinv_Y_h = SYinv_h_max.topLeftCorner(d, d);
                L_inv_times_arrow(Sinv_Y_h, s.segment(idx, d), y.segment(idx, d));
                Sinv_Y_Qyx.middleRows(idx, d) = Sinv_Y_h * Qyx.middleRows(idx, d);
                Sinv_Y_Qyu.middleRows(idx, d) = Sinv_Y_h * Qyu.middleRows(idx, d);
            }

            // less complex, but more slow
            // Eigen::VectorXd Sinv_Y_g = y.head(dim_g).cwiseQuotient(s.head(dim_g));
            // Sinv_Y_Qyx.topRows(dim_g) = Qyx.topRows(dim_g).array().colwise() * Sinv_Y_g.array();
            // Sinv_Y_Qyu.topRows(dim_g) = Qyu.topRows(dim_g).array().colwise() * Sinv_Y_g.array();
            // Eigen::VectorXd Y_Qyx_h_max(ocp->dim_hs_max[k]);
            // Eigen::VectorXd Y_Qyu_h_max(ocp->dim_hs_max[k]);
            // for (int i = 0; i < ocp->dim_hs[k].size(); ++i) {
            //     const int d = ocp->dim_hs[k][i];
            //     const int idx = ocp->dim_hs_top[k][i];
            //     Eigen::Ref<Eigen::VectorXd> Y_Qyx_h = Y_Qyx_h_max.topRows(d);
            //     for (int j = 0; j < dim_rn[k]; ++j) {
            //         L_times_vec(Y_Qyx_h, y.segment(idx, d), Qyx.block(idx, j, d, 1));
            //         L_inv_times_vec(Sinv_Y_Qyx.block(idx, j, d, 1), s.segment(idx, d), Y_Qyx_h);
            //     }
            //     Eigen::Ref<Eigen::VectorXd> Y_Qyu_h = Y_Qyu_h_max.topRows(d);
            //     for (int j = 0; j < ocp->dim_u[k]; ++j) {
            //         L_times_vec(Y_Qyu_h, y.segment(idx, d), Qyu.block(idx, j, d, 1));
            //         L_inv_times_vec(Sinv_Y_Qyu.block(idx, j, d, 1), s.segment(idx, d), Y_Qyu_h);
            //     }
            // }
            
            // Inplace Calculation
            ku_ -= (Qyu.transpose() * Sinv_r); // hat_Qu
            Ku_ -= (Qyx.transpose() * Sinv_Y_Qyu).transpose(); // hat_Qxu
            hat_Quu += (Qyu.transpose() * Sinv_Y_Qyu);
        }
        
        // TODO
        // Equality Constraint
        // if (is_ec_active[k]) {
        // }
        
        Quu_llt.compute(hat_Quu.selfadjointView<Eigen::Upper>());
        if (Quu_llt.info() == Eigen::NumericalIssue) {
            backward_failed = true;
            break;
        }
        
        // ku_ = - Quu_llt.solve(hat_Qu);
        // Ku_ = - Quu_llt.solve(hat_Qxu.transpose());

        // Inplace Calculation
        Quu_llt.solveInPlace(ku_);
        Quu_llt.solveInPlace(Ku_);

        // dV(0) += ku_.transpose() * Qu;
        // dV(1) += 0.5 * ku_.transpose() * Quu * ku_;
        
        Vx = Qx + (Ku_.transpose() * Qu) + (Ku_.transpose() * Quu * ku_) + (Qxu * ku_);
        Vxx = Qxx + (Ku_.transpose() * Qxu.transpose()) + (Qxu * Ku_) + (Ku_.transpose() * Quu * Ku_);
        
        // ku[k] = ku_;
        // Ku[k] = Ku_;

        opterror = std::max({Qu.lpNorm<Eigen::Infinity>(), opterror});

        // Inequality Constraint
        if (is_c_active[k]) {
            Eigen::Ref<Eigen::VectorXd> s = S[k];
            Eigen::Ref<Eigen::VectorXd> y = Y[k];
            Eigen::Ref<Eigen::VectorXd> c_v = C[k];
        
            Eigen::Ref<Eigen::MatrixXd> Qyx = cx_all[k];
            Eigen::Ref<Eigen::MatrixXd> Qyu = cu_all[k];
        
            // const int dim_g = ocp->dim_c[k][0];
            // const int dim_c = ocp->dim_c[k][0] + ocp->dim_c[k][1];
        
            Eigen::Ref<Eigen::VectorXd> ks_ = ks[k];
            Eigen::Ref<Eigen::MatrixXd> Ks_ = Ks[k];
            Eigen::Ref<Eigen::VectorXd> ky_ = ky[k];
            Eigen::Ref<Eigen::MatrixXd> Ky_ = Ky[k];

            ks_ = - (rp + Qyu * ku_);
            Ks_ = - (Qyx + Qyu * Ku_);    

            // more complex, but more fast
            ky_ = Sinv_r + (Sinv_Y_Qyu * ku_);
            Ky_ = Sinv_Y_Qyx + (Sinv_Y_Qyu * Ku_);

            // less complex, but more slow
            // Eigen::VectorXd rd_plus_Y_ds(dim_c);
            // rd_plus_Y_ds.head(dim_g) = y.head(dim_g).cwiseProduct(ks_.head(dim_g));
            // for (int i = 0; i < ocp->dim_hs[k].size(); ++i) {
            //     const int d = ocp->dim_hs[k][i];
            //     const int idx = ocp->dim_hs_top[k][i];
            //     L_times_vec(rd_plus_Y_ds.segment(idx, d), y.segment(idx, d), ks_.segment(idx, d));
            // }
            // rd_plus_Y_ds += rd;

            // ky_.head(dim_g) = rd_plus_Y_ds.head(dim_g).cwiseQuotient(s.head(dim_g));
            // for (int i = 0; i < ocp->dim_hs[k].size(); ++i) {
            //     const int d = ocp->dim_hs[k][i];
            //     const int idx = ocp->dim_hs_top[k][i];
            //     L_inv_times_vec(ky_.segment(idx, d), s.segment(idx, d), rd_plus_Y_ds.segment(idx, d));
            // }
            // ky_ = -ky_;

            // Ky_.topRows(dim_g) = Ks_.topRows(dim_g).array().colwise() * (y.head(dim_g).array() / s.head(dim_g).array());
            // Eigen::VectorXd Y_Ks_h(dim_c);
            // for (int i = 0; i < ocp->dim_hs[k].size(); ++i) {
            //     const int d = ocp->dim_hs[k][i];
            //     const int idx = ocp->dim_hs_top[k][i];
            //     for (int j = 0; j < ocp->dim_x[k]; ++j) {
            //         L_times_vec(Y_Ks_h, y.segment(idx, d), Ks_.block(idx, j, d, 1));
            //         L_inv_times_vec(Ky_.block(idx, j, d, 1), s.segment(idx, d), Y_Ks_h);
            //     }
            // }
            // Ky_ = -Ky_;
            // CHECK: New Value Decrement
            // dV(0) += ky_.transpose() * c_v;
            // dV(1) += ku_.transpose() * Qyu.transpose() * ky_;

            Vx += (Ky_.transpose() * c_v) + (Qyx.transpose() * ky_) + (Ku_.transpose() * Qyu.transpose() * ky_) + (Ky_.transpose() * Qyu * ku_);
            Vxx += (Qyx.transpose() * Ky_) + (Ky_.transpose() * Qyx) + (Ku_.transpose() * Qyu.transpose() * Ky_) + (Ky_.transpose() * Qyu * Ku_);

            // with slack
            // Eigen::VectorXd Qo = Sinv * rd;
            // Eigen::VectorXd Qy = rp;
            // Eigen::MatrixXd I_c = Eigen::VectorXd::Ones(dim_c).asDiagonal();
            // dV(0) += Qo.transpose() * ks_;
            // dV(0) += Qy.transpose() * ky_;
            // dV(1) += ky_.transpose() * Qyu * ku_;
            // dV(1) += ky_.transpose() * I_c * ks_; // Qyy = 0

            // Vx += Ky_.transpose() * Qy + Ku_.transpose() * Qyu.transpose() * ky_ + Ky_.transpose() * Qyu * ku_ + Qyx.transpose() * ky_;
            // Vx += Ks_.transpose() * Qo + Ky_.transpose() * I_c * ks_ + Ks_.transpose() * I_c * ky_;

            // Vxx += Qyx.transpose() * Ky_ + Ky_.transpose() * Qyx + Ky_.transpose() * Qyu * Ku_ + Ku_.transpose() * Qyu.transpose() * Ky_;
            // Vxx += Ky_.transpose() * I_c * Ks_ + Ks_.transpose() * I_c * Ky_;

            // ky[k] = ky_;
            // Ky[k] = Ky_;
            // ks[k] = ks_;
            // Ks[k] = Ks_;

            opterror = std::max({rp.lpNorm<Eigen::Infinity>(), rd.lpNorm<Eigen::Infinity>(), opterror});
            opterror_rp_c = std::max({rp.lpNorm<Eigen::Infinity>(), opterror_rp_c});
            opterror_rd_c = std::max({rd.lpNorm<Eigen::Infinity>(), opterror_rd_c});
        }

        // TODO
        // Equality Constraint
        // if (is_ec_active[k]) {

        // }
    }
    // std::cout << "opterror_rpT_ec = " << opterror_rpT_ec << std::endl;
    // std::cout << "opterror_rdT_ec = " << opterror_rdT_ec << std::endl;
    // std::cout << "opterror_rpT_c = " << opterror_rpT_c << std::endl;
    // std::cout << "opterror_rdT_c = " << opterror_rdT_c << std::endl;
    // std::cout << "opterror_rp_c = " << opterror_rp_c << std::endl;
    // std::cout << "opterror_rd_c = " << opterror_rd_c << std::endl;
}

template <typename Scalar>
void ALIPDDP<Scalar>::checkRegulate() {
    if (forward_failed || backward_failed) {++regulate;}
    // else if (step <= 3) {regulate = regulate;}
    // else {--regulate;}

    if (regulate < 0) {regulate = 0;}
    else if (param.max_regularization < regulate) {regulate = param.max_regularization;}
}

template <typename Scalar>
Eigen::MatrixXd ALIPDDP<Scalar>::L(const Eigen::VectorXd& x) {
    Eigen::MatrixXd Lx = (x(0) * Eigen::VectorXd::Ones(x.rows())).asDiagonal();
    Lx.col(0) = x;
    Lx.row(0) = x.transpose();
    return Lx;
}

template <typename Scalar>
void ALIPDDP<Scalar>::L_inv_times_vec(Eigen::Ref<Eigen::VectorXd> out, const Eigen::Ref<const Eigen::VectorXd>& soc, const Eigen::Ref<const Eigen::VectorXd>& vec) {
    const double& s = soc(0);
    const int n = soc.size() - 1;
    const auto& v = soc.tail(n);
    const double denom = s * s - v.squaredNorm();
    
    out(0) = (s * vec(0) - v.dot(vec.tail(n)))/ denom;
    out.tail(n).noalias() = (- out(0) / s) * v;
    out.tail(n) += (vec.tail(n) / s);
}

template <typename Scalar>
void ALIPDDP<Scalar>::L_times_vec(Eigen::Ref<Eigen::VectorXd> out, const Eigen::Ref<const Eigen::VectorXd>& soc, const Eigen::Ref<const Eigen::VectorXd>& vec) {
    const double& s = soc(0);
    const int n = soc.size() - 1;
    const auto& v = soc.tail(n);
    
    out(0) = s * vec(0) + v.dot(vec.tail(n));
    out.tail(n) = vec(0) * v + s * vec.tail(n);
}

template <typename Scalar>
void ALIPDDP<Scalar>::L_inv(Eigen::Ref<Eigen::MatrixXd> out, const Eigen::Ref<const Eigen::VectorXd>& soc) {
    const double s = soc(0);
    const int n = soc.size() - 1;
    const Eigen::Ref<const Eigen::VectorXd> v = soc.tail(n);
    const double denom = s * s - v.squaredNorm();

    out(0, 0) = s / denom;
    out.block(1,0,n,1) = - v / denom;
    out.block(0,1,1,n) =  out.block(1,0,n,1).transpose();

    out.block(1,1,n,n).noalias() = (v * v.transpose()) / (denom * s);
    out.block(1,1,n,n).diagonal().array() += (1.0 / s);
}

template <typename Scalar>
void ALIPDDP<Scalar>::L_inv_arrow(Eigen::Ref<Eigen::MatrixXd> out, const Eigen::Ref<const Eigen::MatrixXd>& L_inv, const Eigen::Ref<const Eigen::VectorXd>& soc) {
    const double s = soc(0);
    const int n = soc.size() - 1;
    const Eigen::Ref<const Eigen::VectorXd> v = soc.tail(n);

    const double a = L_inv(0, 0);
    const Eigen::Ref<const Eigen::VectorXd> b = L_inv.block(1, 0, n, 1);
    const Eigen::Ref<const Eigen::MatrixXd> C = L_inv.block(1, 1, n, n);

    out(0, 0) = a * s + b.dot(v);
    out.block(0, 1, 1, n).noalias() = (a * v + s * b).transpose();
    out.block(1, 0, n, 1).noalias() = s * b;
    out.block(1, 0, n, 1).noalias() += C * v;

    auto out_sub = out.block(1, 1, n, n);
    out_sub.noalias()  = s * C;
    out_sub.noalias() += b * v.transpose();
}

template <typename Scalar>
void ALIPDDP<Scalar>::L_inv_times_arrow(Eigen::Ref<Eigen::MatrixXd> out, const Eigen::Ref<const Eigen::VectorXd>& soc_inv, const Eigen::Ref<const Eigen::VectorXd>& soc_arrow) {
    const int n = soc_inv.size() - 1;

    const double& s = soc_inv(0);
    const auto& v = soc_inv.tail(n);
    
    const double& a = soc_arrow(0);
    const auto& b = soc_arrow.tail(n);
    
    const double inv_s = 1.0 / s;
    const double denom = s * s - v.squaredNorm();
    const double inv_denom = 1.0 / denom;

    const Eigen::VectorXd m_v_s = - v * inv_s;

    out(0, 0) = (s * a - v.dot(b)) * inv_denom;
    out.block(1, 0, n, 1).noalias() = m_v_s * out(0, 0);
    out.block(0, 1, 1, n) = (s * b.transpose() - a * v.transpose()) * inv_denom;
    out.block(1, 1, n, n).noalias() = m_v_s * out.block(0, 1, 1, n);
    out.block(1, 1, n, n).diagonal().array() += a * inv_s;
}

template <typename Scalar>
Eigen::VectorXd ALIPDDP<Scalar>::perturb(const Eigen::VectorXd& xn, const Eigen::VectorXd& x) {
    // if (this->param.is_quaternion_in_state) {
    //     // Eigen::VectorXd dx(dim_rn);
    //     // Eigen::VectorXd q_qn = Lq(x.segment(q_idx, q_dim)).cast<double>().transpose() * xn.segment(q_idx, q_dim);
    //     // dx << xn.segment(0,q_idx) - x.segment(0,q_idx),
    //     //     q_qn.segment(1,3)/q_qn(0);
    //     // return dx;
    // }
    // else {
    //     return xn - x;
    // } 
    return xn - x;
}

template <typename Scalar>
void ALIPDDP<Scalar>::forwardPass() {
    Eigen::VectorXd dx;
    std::vector<Eigen::VectorXd> X_new(ocp->N+1);
    std::vector<Eigen::VectorXd> U_new(ocp->N);
    std::vector<Eigen::VectorXd> S_new(ocp->N);
    std::vector<Eigen::VectorXd> Y_new(ocp->N);
    std::vector<Eigen::VectorXd> C_new(ocp->N);
    std::vector<Eigen::VectorXd> R_new(ocp->N);
    std::vector<Eigen::VectorXd> Z_new(ocp->N);
    std::vector<Eigen::VectorXd> EC_new(ocp->N);

    Eigen::VectorXd dxT;
    Eigen::VectorXd ST_new;
    Eigen::VectorXd YT_new;
    Eigen::VectorXd CT_new;
    Eigen::VectorXd RT_new;
    Eigen::VectorXd ZT_new;
    Eigen::VectorXd ECT_new;

    // double tau = std::max(0.99, 1.0 - param.mu);
    double tau = 0.9;
    const double one_tau = 1.0 - tau;
    
    double cost_new = 0.0;
    double logcost_new = 0.0;
    double barriercost_new = 0.0;
    double barriercostT_new = 0.0;
    double alcost_new = 0.0;
    double alcostT_new = 0.0;
    double error_new = 0.0;

    double dV_act;
    double dV_exp;

    for (step = 0; step < this->param.max_step_iter; ++step) {

        forward_failed = 0;
        const double step_size = step_list[step];

        // dV_exp = -(step_size * dV(0) + step_size * step_size * dV(1));
        // CHECK: Using Expected Value Decrement -> For Early Termination
        if (param.forward_early_termination) {
            if (error <= param.tolerance && dV_exp > 0) {
                forward_failed = 3; continue;
            }
        }

        X_new[0] = X[0];
        for (int k = 0; k < ocp->N; ++k) {
            dx = perturb(X_new[k], X[k]);
            U_new[k] = U[k] + (step_size * ku[k]) + Ku[k] * dx;
            X_new[k+1] = ocp->dynamics_seq[k]->f(X_new[k], U_new[k]);
            if (is_c_active[k]) {
                const int dim_g = ocp->dim_c[k][0];
                S_new[k] = S[k] + (step_size * ks[k]) + Ks[k] * dx;
                Y_new[k] = Y[k] + (step_size * ky[k]) + Ky[k] * dx;
                if (dim_g) {
                    const int d = dim_g;
                    const Eigen::Ref<const Eigen::VectorXd> S_new_head = S_new[k].head(d);
                    const Eigen::Ref<const Eigen::VectorXd> Y_new_head = Y_new[k].head(d);
                    const Eigen::Ref<const Eigen::VectorXd> S_head = S[k].head(d);
                    const Eigen::Ref<const Eigen::VectorXd> Y_head = Y[k].head(d);
                    if ((S_new_head.array() < (one_tau) * S_head.array()).any() || (Y_new_head.array() < (one_tau) * Y_head.array()).any()) {
                        forward_failed = 11; break;
                    }
                }
                for (int i = 0; i < ocp->dim_hs[k].size(); ++i) {
                    const int n = ocp->dim_hs[k][i] - 1;
                    const int idx = ocp->dim_hs_top[k][i];
                    const double S_new_norm = S_new[k](idx) - S_new[k].segment(idx + 1, n).norm();
                    const double Y_new_norm = Y_new[k](idx) - Y_new[k].segment(idx + 1, n).norm();
                    const double S_norm = (one_tau) * (S[k](idx) - S[k].segment(idx + 1, n).norm());
                    const double Y_norm = (one_tau) * (Y[k](idx) - Y[k].segment(idx + 1, n).norm());
                    if (S_new_norm < S_norm || Y_new_norm < Y_norm) {
                        forward_failed = 13; break;
                    }
                }
                if (forward_failed) {break;}
                
                C_new[k] = ocp->c[k](X_new[k], U_new[k]);
            }
            if (is_ec_active[k]) {
                R_new[k] = R[k] + (step_size * kr[k]) + Kr[k] * dx;
                Z_new[k] = Z[k] + (step_size * kz[k]) + Kz[k] * dx;

                EC_new[k] = ocp->ec[k](X_new[k], U_new[k]);
            }        
        }
        if (forward_failed) {continue;}
        
        dxT = perturb(X_new[ocp->N], X[ocp->N]);
        if (is_cT_active) {
            ST_new = ST + (step_size * ksT) + KsT * dxT;
            YT_new = YT + (step_size * kyT) + KyT * dxT;
            const int dim_gT = ocp->dim_cT[0];
            if (dim_gT) {
                const int d = dim_gT;
                const Eigen::Ref<const Eigen::VectorXd> ST_new_head = ST_new.head(d);
                const Eigen::Ref<const Eigen::VectorXd> YT_new_head = YT_new.head(d);
                const Eigen::Ref<const Eigen::VectorXd> ST_head = ST.head(d);
                const Eigen::Ref<const Eigen::VectorXd> YT_head = YT.head(d);
                if ((ST_new_head.array() < (one_tau) * ST_head.array()).any() || (YT_new_head.array() < (one_tau) * YT_head.array()).any()) {
                    forward_failed = 21; continue;
                }
            }
            for (int i = 0; i < ocp->dim_hTs.size(); ++i) {
                const int n = ocp->dim_hTs[i] - 1;
                const int idx = ocp->dim_hTs_top[i];
                const double ST_new_norm = ST_new(idx) - ST_new.segment(idx + 1, n).norm();
                const double YT_new_norm = YT_new(idx) - YT_new.segment(idx + 1, n).norm();
                const double ST_norm = (one_tau) * (ST(idx) - ST.segment(idx + 1, n).norm());
                const double YT_norm = (one_tau) * (YT(idx) - YT.segment(idx + 1, n).norm());
                if (ST_new_norm < ST_norm || YT_new_norm < YT_norm) {
                    forward_failed = 23; break;
                }
            }
            if (forward_failed) {continue;}

            CT_new = ocp->cT(X_new[ocp->N]);
        }

        if (is_ecT_active) {
            RT_new = RT + (step_size * krT) + KrT * dxT;
            ZT_new = ZT + (step_size * kzT) + KzT * dxT;

            ECT_new = ocp->ecT(X_new[ocp->N]);
        }
        
        error_new = 0.0;
        for (int k = 0; k < ocp->N; ++k) {
            if (is_c_active[k]) {
                error_new += (C_new[k] + S_new[k]).array().abs().sum();
            }
            if (is_ec_active[k]) {
                error_new += (EC_new[k] + R_new[k]).array().abs().sum();
            }
        }
        if (is_ecT_active) {
            error_new += (ECT_new + RT_new).array().abs().sum();
        }
        if (is_cT_active) {
            error_new += (CT_new + ST_new).array().abs().sum();
        }
        error_new = std::max(param.tolerance, error_new);

        // Cost
        barriercost_new = 0.0;
        barriercostT_new = 0.0;
        alcost_new = 0.0;
        alcostT_new = 0.0;
        cost_new = calculateTotalCost(X_new, U_new);

        for (int k = 0; k < ocp->N; ++k) {
            const int dim_g = ocp->dim_c[k][0];
            if (dim_g) {
                barriercost_new += S_new[k].head(dim_g).array().log().sum();
            }
            for (int i = 0; i < ocp->dim_hs[k].size(); ++i) {
                const int d = ocp->dim_hs[k][i] - 1;
                const int idx = ocp->dim_hs_top[k][i];
                const double s = S_new[k](idx);
                const double v_2 = S_new[k].segment(idx + 1, d).squaredNorm();
                barriercost_new += 0.5 * log(s * s - v_2);
            }
        }
        if (ocp->dim_cT[0] > 0) {
            barriercostT_new += ST_new.head(ocp->dim_cT[0]).array().log().sum();
        }
        for (int i = 0; i < ocp->dim_hTs.size(); ++i) {
            const int d   = ocp->dim_hTs[i];
            const int idx = ocp->dim_hTs_top[i];
            const double s = ST_new(idx);
            const double v_2 = ST_new.segment(idx+1, d-1).squaredNorm();
            barriercostT_new += 0.5 * log(s * s - v_2);
        }
        for (int k = 0; k < ocp->N; ++k) {
            if (is_ec_active[k]) {
                alcost_new += lambda[k].transpose() * R_new[k] + 0.5 * param.rho * R_new[k].squaredNorm();
            }
        }
        if (is_ecT_active) {
            alcostT_new += lambdaT.transpose() * RT_new + 0.5 * param.rho * RT_new.squaredNorm();
        }
        logcost_new = cost_new - (param.mu * barriercost_new + param.muT * barriercostT_new) + (alcost_new + alcostT_new);
        if (std::isnan(logcost_new)) {forward_failed = 5; continue;}
        dV_act = logcost - logcost_new;
                
        // Error Decrement
        if (param.forward_filter == 0) {
            if (error < error_new) {forward_failed = 1; continue;}
            if (dV_act < 0.0) {forward_failed = 2; continue;}
        }
        if (param.forward_filter == 1) {
            if (error <= param.tolerance) {
                if (error < error_new) {forward_failed = 1; continue;}
            }
            if (error <= error_new) {forward_failed = 1;}
            if (forward_failed == 1) {
                if (dV_act < -(param.forward_cost_threshold * error)) {forward_failed = 2; continue;}
                else {forward_failed = 0;}
            }
        }

        if (param.forward_early_termination) {
            if (error <= param.tolerance){
                // if (dV_act < 0.0) {forward_failed = 2; continue;}
                // if (dV_exp <= 0.0) {forward_failed = 3; continue;}
    
                if (dV_exp >= 0.0) {
                    if (!(1e-4 * dV_exp < dV_act && dV_act < 10 * dV_exp)) {forward_failed = 4; continue;}
                }
            }
        }
        
        if (!forward_failed) {break;}
    }

    if (!forward_failed) {
        cost = cost_new;
        logcost = logcost_new;
        error = error_new;
        X = std::move(X_new);
        U = std::move(U_new);
        if (is_c_active_all) {
            S = std::move(S_new);
            Y = std::move(Y_new);
            C = std::move(C_new);
        }
        if (is_ec_active_all) {
            R = std::move(R_new);
            Z = std::move(Z_new);
            EC = std::move(EC_new);
        }
        if (is_cT_active) {
            ST = std::move(ST_new);
            YT = std::move(YT_new);
            CT = std::move(CT_new);
        }
        if (is_ecT_active) {
            RT = std::move(RT_new);
            ZT = std::move(ZT_new);
            ECT = std::move(ECT_new);
        }
    }
    // else {std::cout<<"Forward Failed"<<std::endl;}
}

template <typename Scalar>
std::vector<Eigen::VectorXd> ALIPDDP<Scalar>::getResX() {
    return X;
}

template <typename Scalar>
std::vector<Eigen::VectorXd> ALIPDDP<Scalar>::getResU() {
    return U;
}

template <typename Scalar>
std::vector<double> ALIPDDP<Scalar>::getAllCost() {
    return all_cost;
}

template <typename Scalar>
void ALIPDDP<Scalar>::logPrint() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(4) << iter
              << std::setw(4) << inner_iter
              << std::setw(3) << backward_failed
              << std::setw(3) << forward_failed
              << std::setw(7) << param.mu
              << std::setw(16) << param.rho
              << std::setw(25) << logcost
              << std::setw(22) << opterror
              << std::setw(18) << error
              << std::setw(4) << regulate
              << std::setw(7) << step_list[step]
              << std::setw(5) << update_counter << std::endl;
}