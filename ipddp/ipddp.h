#pragma once

#include "param.h"
#include "model_base.h"
#include "quat_model_base.h"
// #include "helper_function.h"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <functional>
#include <cmath>
#include <ctime>

#include <iomanip>
#include <iostream>
#include <memory>
#include <type_traits>

class IPDDP {
public:
    template<typename ModelClass>
    explicit IPDDP(std::shared_ptr<ModelClass> model_ptr);
    ~IPDDP();

    void init(Param param);
    void solve();

    Eigen::MatrixXd getInitX();
    Eigen::MatrixXd getInitU();
    Eigen::MatrixXd getResX();
    Eigen::MatrixXd getResU();
    std::vector<double> getAllCost();

private:
    std::shared_ptr<ModelBase> model;

    // Constraint Stack
    std::vector<int> dim_hs_top; // Connic Constraint Head Stack
    std::vector<int> dim_hTs_top; // Connic Constraint Head Stack (Terminal)
    int dim_hs_max; // Maximum Dimension of Connic Constraint
    int dim_hTs_max; // Maximum Dimension of Connic Constraint (Terminal)

    Eigen::MatrixXd X; // State
    Eigen::MatrixXd U; // Input
    
    Eigen::MatrixXd Z; // Equality Lagrangian Multiplier
    Eigen::MatrixXd R; // Equality Slack
    Eigen::MatrixXd Y; // Inequality Lagrangian Multiplier
    Eigen::MatrixXd S; // Inequality Slack
    Eigen::MatrixXd C; // Inequality Constraint
    Eigen::MatrixXd EC; // Equality Constraint

    Eigen::VectorXd ZT; // Equality Lagrangian Multiplier (Terminal)
    Eigen::VectorXd RT; // Equality Slack (Terminal)
    Eigen::VectorXd YT; // Inequality Lagrangian Multiplier (Terminal)
    Eigen::VectorXd ST; // Inequality Slack (Terminal)
    Eigen::VectorXd CT; // Inequality Constraint (Terminal)
    Eigen::VectorXd ECT; // Equality Constraint (Terminal)

    Eigen::VectorXd scale_ec;
    Eigen::VectorXd scale_c;
    Eigen::VectorXd scale_ecT;
    Eigen::VectorXd scale_cT;
    
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

    Eigen::MatrixXd fx_all;
    Eigen::MatrixXd fu_all;
    // Eigen::MatrixXd fxx_all;
    // Eigen::MatrixXd fxu_all;
    // Eigen::MatrixXd fuu_all;
    Eigen::VectorXd px_all;
    Eigen::MatrixXd pxx_all;
    Eigen::MatrixXd qx_all;
    Eigen::MatrixXd qu_all;
    Eigen::MatrixXd qdd_all;
    Eigen::MatrixXd cx_all;
    Eigen::MatrixXd cu_all;
    Eigen::MatrixXd ecx_all;
    Eigen::MatrixXd ecu_all;
    Eigen::MatrixXd cTx_all;
    Eigen::MatrixXd ecTx_all;

    Eigen::MatrixXd ku; // Input Feedforward Gain 
    Eigen::MatrixXd kr; // Equality Slack Feedforward Gain
    Eigen::MatrixXd kz; // Equality Lagrangian Multiplier Feedforward Gain
    Eigen::MatrixXd ks; // Inequality Slack Feedforward Gain
    Eigen::MatrixXd ky; // Inequality Lagrangian Multiplier Feedforward Gain
    
    Eigen::MatrixXd Ku; // Input Feedback Gain
    Eigen::MatrixXd Kr; // Equality Slack Feedback Gain
    Eigen::MatrixXd Kz; // Equality Lagrangian Multiplier Feedback Gain
    Eigen::MatrixXd Ks; // Inequality Slack Feedback Gain
    Eigen::MatrixXd Ky; // Inequality Lagrangian Multiplier Feedback Gain

    Eigen::VectorXd krT; // Equality Slack Feedforward Gain (Terminal)
    Eigen::VectorXd kzT; // Equality Lagrangian Multiplier Feedforward Gain (Terminal)
    Eigen::MatrixXd KrT; // Equality Slack Feedback Gain (Terminal)
    Eigen::MatrixXd KzT; // Equality Lagrangian Multiplier Feedback Gain (Terminal)

    Eigen::VectorXd ksT; // Inequality Slack Feedforward Gain (Terminal)
    Eigen::VectorXd kyT; // Inequality Lagrangian Multiplier Feedforward Gain (Terminal)
    Eigen::MatrixXd KsT; // Inequality Slack Feedback Gain (Terminal)
    Eigen::MatrixXd KyT; // Inequality Lagrangian Multiplier Feedback Gain (Terminal)

    Eigen::VectorXd e;
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
    double calculateTotalCost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U);
    void logPrint();
};

template<typename ModelClass>
IPDDP::IPDDP(std::shared_ptr<ModelClass> model_ptr) : model(model_ptr) {
    { // TODO: Move to Model
        static_assert(std::is_base_of<ModelBase, ModelClass>::value, "ModelClass must be derived from ModelBase");
        if (std::is_base_of<QuatModelBase, ModelClass>::value) {
            model->dim_rn = model->dim_x - 1;
        }
        else {model->dim_rn = model->dim_x;}

        // Inequality Constraint Stack
        model->dim_c = model->dim_g + accumulate(model->dim_hs.begin(), model->dim_hs.end(), 0);
        int dim_h_top = model->dim_g;
        for (auto dim_h : model->dim_hs) {
            dim_hs_top.push_back(dim_h_top);
            dim_h_top += dim_h;
        }
        model->c = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
            VectorXdual2nd c_n(model->dim_c);
            if (model->dim_g) {
                c_n.topRows(model->dim_g) = model->g(x, u);
            }
            for (int i = 0; i < model->dim_hs.size(); ++i) {
                c_n.middleRows(dim_hs_top[i], model->dim_hs[i]) = model->hs[i](x, u);
            }
            return c_n;
        };
        if (model->dim_c - model->dim_g != 0) {
            dim_hs_max = *std::max_element(model->dim_hs.begin(), model->dim_hs.end());
        }
        // Inequality Constraint Stack (Terminal)
        model->dim_cT = model->dim_gT + accumulate(model->dim_hTs.begin(), model->dim_hTs.end(), 0);
        int dim_hT_top = model->dim_gT;
        for (auto dim_hT : model->dim_hTs) {
            dim_hTs_top.push_back(dim_hT_top);
            dim_hT_top += dim_hT;
        }
        model->cT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
            VectorXdual2nd cT_n(model->dim_cT);
            if (model->dim_gT) {
                cT_n.topRows(model->dim_gT) = model->gT(x);
            }
            for (int i = 0; i < model->dim_hTs.size(); ++i) {
                cT_n.middleRows(dim_hTs_top[i], model->dim_hTs[i]) = model->hTs[i](x);
            }
            return cT_n;
        };
        if (model->dim_cT - model->dim_gT != 0) {
            dim_hTs_max = *std::max_element(model->dim_hTs.begin(), model->dim_hTs.end());
        }
    } // TODO: Move to Model


    // Initialization
    if (model->X_init.size()) {X = model->X_init;}
    else {X = Eigen::MatrixXd::Zero(model->dim_x, model->N+1);}
    if (model->U_init.size()) {U = model->U_init;}
    else {U = Eigen::MatrixXd::Zero(model->dim_u, model->N);}

    // CHECK!!!
    if (model->R_init.size()) {R = model->R_init;}
    else {R = Eigen::MatrixXd::Zero(model->dim_ec, model->N);}
    if (model->Z_init.size()) {Z = model->Z_init;}
    else {Z = Eigen::MatrixXd::Zero(model->dim_ec, model->N);}


    if (model->S_init.size()) {S = model->S_init;}
    else {
        S = Eigen::MatrixXd::Zero(model->dim_c, model->N);
        if (model->dim_g) {S.topRows(model->dim_g) = Eigen::MatrixXd::Ones(model->dim_g, model->N);}
        for (auto dim_h_top : dim_hs_top) {S.row(dim_h_top) = Eigen::VectorXd::Ones(model->N);}
    }
    if (model->Y_init.size()) {Y = model->Y_init;}
    else {
        Y = Eigen::MatrixXd::Zero(model->dim_c, model->N);
        if (model->dim_g) {Y.topRows(model->dim_g) = Eigen::MatrixXd::Ones(model->dim_g, model->N);}
        for (auto dim_h_top : dim_hs_top) {Y.row(dim_h_top) = Eigen::VectorXd::Ones(model->N);}
    }

    if (model->RT_init.size()) {RT = model->RT_init;}
    else {RT = Eigen::VectorXd::Zero(model->dim_ecT);}
    if (model->ZT_init.size()) {ZT = model->ZT_init;}
    else {ZT = Eigen::VectorXd::Zero(model->dim_ecT);}

    if (model->ST_init.size()) {ST = model->ST_init;}
    else {
        ST = Eigen::VectorXd::Zero(model->dim_cT);
        if (model->dim_gT) {ST.topRows(model->dim_gT) = Eigen::VectorXd::Ones(model->dim_gT);}
        for (auto dim_hT_top : dim_hTs_top) {ST(dim_hT_top) = 1.0;}
    }
    if (model->YT_init.size()) {YT = model->YT_init;}
    else {
        YT = Eigen::VectorXd::Zero(model->dim_cT);
        if (model->dim_gT) {YT.topRows(model->dim_gT) = Eigen::VectorXd::Ones(model->dim_gT);}
        for (auto dim_hT_top : dim_hTs_top) {YT(dim_hT_top) = 1.0;}
    }

    fx_all.resize(model->dim_rn, model->dim_rn * model->N);
    fu_all.resize(model->dim_rn, model->dim_u * model->N);
    px_all.resize(model->dim_rn);
    pxx_all.resize(model->dim_rn, model->dim_rn);
    qx_all.resize(model->dim_rn, model->N);
    qu_all.resize(model->dim_u, model->N);
    qdd_all.resize(model->dim_rn + model->dim_u, (model->dim_rn + model->dim_u) * model->N);
    // if (DDP) {
    //     fxx_all.resize(model->dim_rn, model->dim_rn * model->dim_rn * model->N);
    //     fxu_all.resize(model->dim_rn, model->dim_rn * model->dim_u * model->N);
    //     fuu_all.resize(model->dim_rn, model->dim_u * model->dim_u * model->N);
    // }
    if (model->dim_c) {
        cx_all.resize(model->dim_c, model->dim_rn * model->N);
        cu_all.resize(model->dim_c, model->dim_u * model->N);
        scale_c = Eigen::VectorXd::Ones(model->dim_c);
    }
    if (model->dim_ec) {
        ecx_all.resize(model->dim_ec, model->dim_rn * model->N);
        ecu_all.resize(model->dim_ec, model->dim_u * model->N);
        scale_ec = Eigen::VectorXd::Ones(model->dim_ec);
    }
    if (model->dim_cT) {
        cTx_all.resize(model->dim_cT, model->dim_rn);
        scale_cT = Eigen::VectorXd::Ones(model->dim_cT);
    }
    if (model->dim_ecT) {
        ecTx_all.resize(model->dim_ecT, model->dim_rn);
        scale_ecT = Eigen::VectorXd::Ones(model->dim_ecT);
    }

    ku.resize(model->dim_u, model->N);
    kr.resize(model->dim_ec, model->N);
    kz.resize(model->dim_ec, model->N);
    ks.resize(model->dim_c, model->N);
    ky.resize(model->dim_c, model->N);
    Ku.resize(model->dim_u, model->dim_rn * model->N);
    Kr.resize(model->dim_ec, model->dim_rn * model->N);
    Kz.resize(model->dim_ec, model->dim_rn * model->N);
    Ks.resize(model->dim_c, model->dim_rn * model->N);
    Ky.resize(model->dim_c, model->dim_rn * model->N);

    krT.resize(model->dim_ecT);
    kzT.resize(model->dim_ecT);
    ksT.resize(model->dim_cT);
    kyT.resize(model->dim_cT);
    KrT.resize(model->dim_ecT, model->dim_rn);
    KzT.resize(model->dim_ecT, model->dim_rn);
    KsT.resize(model->dim_cT, model->dim_rn);
    KyT.resize(model->dim_cT, model->dim_rn);

    e = Eigen::VectorXd::Ones(model->dim_c);
    for (int i = 0; i < model->dim_hs.size(); ++i) {
        e.middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1) = Eigen::VectorXd::Zero(model->dim_hs[i]-1);
    }
    eT = Eigen::VectorXd::Ones(model->dim_cT);
    for (int i = 0; i < model->dim_hTs.size(); ++i) {
        eT.middleRows(dim_hTs_top[i]+1, model->dim_hTs[i]-1) = Eigen::VectorXd::Zero(model->dim_hTs[i]-1);
    }
}

IPDDP::~IPDDP() {
}

void IPDDP::init(Param param) {
    this->param = param;

    if (!this->param.lambda.size()) {this->param.lambda = Eigen::VectorXd::Zero(model->dim_ec);}
    if (!this->param.lambdaT.size()) {this->param.lambdaT = Eigen::VectorXd::Zero(model->dim_ecT);}

    initialRoll();
    if (this->param.mu == 0) {this->param.mu = cost / model->N / model->dim_c;} // Auto Select

    // TEST
    // if (model->dim_ecT) {this->param.lambdaT = this->param.lambdaT + param.rho * RT;}

    resetFilter();
    resetRegulation();

    for (int i = 0; i <= this->param.max_step_iter; ++i) {
        step_list.push_back(std::pow(2.0, static_cast<double>(-i)));
    }
}

void IPDDP::initialRoll() {
    if (model->dim_c) {C.resize(model->dim_c, model->N);}
    if (model->dim_ec) {EC.resize(model->dim_ec, model->N);}

    for (int t = 0; t < model->N; ++t) {
        const Eigen::VectorXd& xt = X.col(t);
        const Eigen::VectorXd& ut = U.col(t);
        if (model->dim_c) {C.col(t) = model->c(xt, ut).cast<double>();}
        if (model->dim_ec) {EC.col(t) = model->ec(xt, ut).cast<double>();}
        X.col(t+1) = model->f(xt, ut).cast<double>();
    }
    if (model->dim_cT) {CT = model->cT(X.col(model->N)).cast<double>();}
    if (model->dim_ecT) {ECT = model->ecT(X.col(model->N)).cast<double>();}

    cost = calculateTotalCost(X, U);
}

void IPDDP::resetFilter() {
    double barriercost = 0.0;
    double barriercostT = 0.0;
    double alcost = 0.0;
    double alcostT = 0.0;

    if (model->dim_g) {barriercost += S.topRows(model->dim_g).array().log().sum();}
    for (int i = 0; i < model->dim_hs.size(); ++i) {barriercost += log(S.row(dim_hs_top[i]).array().pow(2.0).sum() - S.middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).array().pow(2.0).sum())/2;}
    if (model->dim_gT) {barriercostT += ST.topRows(model->dim_gT).array().log().sum();}
    for (int i = 0; i < model->dim_hTs.size(); ++i) {barriercostT += log(ST.row(dim_hTs_top[i]).array().pow(2.0).sum() - ST.middleRows(dim_hTs_top[i]+1, model->dim_hTs[i]-1).array().pow(2.0).sum())/2;}

    // CHECK
    if (model->dim_ec) {alcost += (param.lambda.transpose() * R).sum() + (0.5 * param.rho * R.squaredNorm());}
    if (model->dim_ecT) {alcostT += (param.lambdaT.transpose() * RT) + (0.5 * param.rho * RT.squaredNorm());}

    logcost = cost - (param.mu * barriercost + param.muT * barriercostT) + (alcost + alcostT);

    error = 0.0;
    if (model->dim_ec) {error += (EC + R).array().abs().sum();}
    if (model->dim_c) {error += (C + S).array().abs().sum();}
    if (model->dim_ecT) {error += (ECT + RT).array().abs().sum();}
    if (model->dim_cT) {error += (CT + ST).array().abs().sum();}
    error = std::max(param.tolerance, error);
    
    step = 0;
    forward_failed = false;
}

void IPDDP::resetRegulation() {
    this->regulate = 0;
    this->backward_failed = false;
}

double IPDDP::calculateTotalCost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U) {
    dual2nd cost = 0.0;
    for (int t = 0; t < model->N; ++t) {
        cost += model->q(X.col(t), U.col(t));
    }
    cost += model->p(X.col(model->N));
    return static_cast<double>(cost.val);
}

void IPDDP::solve() {
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
                bool updated = false;
                if (model->dim_c && opterror_rp_c < param.tolerance && opterror_rd_c < param.tolerance) {
                    if (param.mu > param.mu_min) {updated = true;}
                    param.mu = std::max(param.mu_min, std::min(param.mu_mul * param.mu, std::pow(param.mu, param.mu_exp)));
                }
                if (model->dim_cT && opterror_rpT_c < param.tolerance && opterror_rdT_c < param.tolerance) {
                    if (param.muT > param.mu_min) {updated = true;}
                    param.muT = std::max(param.mu_min, std::min(param.mu_mul * param.muT, std::pow(param.muT, param.mu_exp)));
                }
                if (model->dim_ecT && opterror_rpT_ec < param.tolerance && opterror_rdT_ec < param.tolerance) {
                    if (param.rho < param.rho_max) {updated = true;}
                    param.rho = std::min(param.rho_max, param.rho_mul * param.rho);
                    param.lambdaT = param.lambdaT + param.rho * RT;
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

        // CHECK
        if ((!(model->dim_c) || param.mu <= param.mu_min)
        && (!(model->dim_cT) || param.muT <= param.mu_min)
        && (!(model->dim_ec || model->dim_ecT) || (param.rho_max <= param.rho))
        && (model->dim_c || model->dim_cT || model->dim_ec || model->dim_ecT)) {
            std::cout << "Outer Max/Min" << std::endl;
            return;
        }

        // Update Outer Loop Parameters
        if (model->dim_c) {param.mu = std::max(param.mu_min, std::min(param.mu_mul * param.mu, std::pow(param.mu, param.mu_exp)));}
        if (model->dim_cT) {param.muT = std::max(param.mu_min, std::min(param.mu_mul * param.muT, std::pow(param.muT, param.mu_exp)));}
        // param.lambdaT = param.lambdaT + param.rho * RT;
        // CHECK
        // if (model->dim_ec || model->dim_ecT) {param.rho = std::min(param.rho_max, std::max(param.rho_mul * param.rho, 1.0 / param.mu));}
        if (model->dim_ec || model->dim_ecT) {param.rho = std::min(param.rho_max, param.rho_mul * param.rho);}
        if (model->dim_ecT) {param.lambdaT = param.lambdaT + param.rho * RT;}
        resetFilter();
        resetRegulation();
    }
}

void IPDDP::calculateAllDiff() {
    // CHECK: Multithreading (TODO: with CUDA)
    // CHECK 1: Making branch in for loop is fine for parallelization?
    // CHECK 2: Move to Model to make solver only consider Eigen (not autodiff::dual)

    // #pragma omp parallel for
    for (int t = 0; t < model->N; ++t) {
        VectorXdual2nd x = X.col(t).cast<dual2nd>();
        VectorXdual2nd u = U.col(t).cast<dual2nd>();

        const int t_dim_rn = t * model->dim_rn;
        const int t_dim_u = t * model->dim_u;

        fx_all.middleCols(t_dim_rn, model->dim_rn) = model->fx(x,u);
        fu_all.middleCols(t_dim_u, model->dim_u) = model->fu(x,u);
        qx_all.col(t) = model->qx(x,u);
        qu_all.col(t) = model->qu(x,u);
        qdd_all.middleCols(t_dim_rn + t_dim_u, model->dim_rn + model->dim_u) = model->qdd(x,u);
        // if (DDP) {
        //     fxx_all.middleCols(t_dim_rn*model->dim_rn, model->dim_rn*model->dim_rn) = model->fxx(x,u);
        //     fxu_all.middleCols(t_dim_rn*model->dim_u, model->dim_rn*model->dim_u) = model->fxu(x,u);
        //     fuu_all.middleCols(t_dim_u*model->dim_u, model->dim_u*model->dim_u) = model->fuu(x,u);
        // }
        if (model->dim_c) {
            cx_all.middleCols(t_dim_rn, model->dim_rn) = model->cx(x,u);
            cu_all.middleCols(t_dim_u, model->dim_u) = model->cu(x,u);
        }
        if (model->dim_ec) {
            ecx_all.middleCols(t_dim_rn, model->dim_rn) = model->ecx(x,u);
            ecu_all.middleCols(t_dim_u, model->dim_u) = model->ecu(x,u);
        }
    }
    VectorXdual2nd xT = X.col(model->N).cast<dual2nd>();
    px_all = model->px(xT);
    pxx_all = model->pxx(xT);
    if (model->dim_cT) {cTx_all = model->cTx(xT);}
    if (model->dim_ecT) {ecTx_all = model->ecTx(xT);}
}

void IPDDP::backwardPass() {
    Eigen::VectorXd Vx(model->dim_rn);
    Eigen::MatrixXd Vxx(model->dim_rn, model->dim_rn);
    Eigen::MatrixXd Vxx_reg1(model->dim_rn, model->dim_rn);

    Eigen::VectorXd Qx(model->dim_rn), Qu(model->dim_u);
    Eigen::MatrixXd Qxx(model->dim_rn,model->dim_rn), Qxu(model->dim_rn,model->dim_u), Quu(model->dim_u,model->dim_u);
    Eigen::MatrixXd Quu_sim(model->dim_u,model->dim_u);

    // Eigen::VectorXd hat_Qu(model->dim_u);
    // Eigen::MatrixXd hat_Qxu(model->dim_rn,model->dim_u)
    Eigen::MatrixXd hat_Quu(model->dim_u,model->dim_u);

    Eigen::VectorXd rp(model->dim_c);
    Eigen::VectorXd rd(model->dim_c);
    
    Eigen::VectorXd Sinv_r(model->dim_c);
    Eigen::MatrixXd Sinv_Y_Qyx(model->dim_c, model->dim_rn);
    Eigen::MatrixXd Sinv_Y_Qyu(model->dim_c, model->dim_u);
    
    Eigen::LLT<Eigen::MatrixXd> Quu_llt(model->dim_u);

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
    if (model->dim_cT) {
        Eigen::Ref<const Eigen::MatrixXd> QyxT = cTx_all;

        Eigen::VectorXd rpT = CT + ST;
        Eigen::VectorXd rdT(model->dim_cT);
        rdT.head(model->dim_gT) = ST.head(model->dim_gT).cwiseProduct(YT.head(model->dim_gT));
        for (int i = 0; i < model->dim_hTs.size(); ++i) {
            const int idx = dim_hTs_top[i];
            const int n = model->dim_hTs[i];
            L_times_vec(rdT.segment(idx, n), YT.segment(idx, n), ST.segment(idx, n));
        }
        rdT -= param.muT * eT;
        Eigen::VectorXd rT(model->dim_cT);
        rT.head(model->dim_gT) = YT.head(model->dim_gT).cwiseProduct(rpT.head(model->dim_gT));
        for (int i = 0; i < model->dim_hTs.size(); ++i) {
            const int d = model->dim_hTs[i];
            const int idx = dim_hTs_top[i];
            L_times_vec(rT.segment(idx, d), YT.segment(idx, d), rpT.segment(idx, d));
        }
        rT -= rdT;

        Eigen::VectorXd Sinv_rT(model->dim_cT);
        Sinv_rT.head(model->dim_gT) = rT.head(model->dim_gT).cwiseQuotient(ST.head(model->dim_gT));
        for (int i = 0; i < model->dim_hTs.size(); ++i) {
            const int d = model->dim_hTs[i];
            const int idx = dim_hTs_top[i];
            L_inv_times_vec(Sinv_rT.segment(idx, d), ST.segment(idx, d), rT.segment(idx, d));
        }
        
        Eigen::MatrixXd Sinv_Y_QyxT(model->dim_cT, model->dim_rn);
        Sinv_Y_QyxT.topRows(model->dim_gT) = QyxT.topRows(model->dim_gT).array().colwise() * (YT.head(model->dim_gT).array() / ST.head(model->dim_gT).array());
        Eigen::MatrixXd Sinv_Y_hT_max(dim_hTs_max, dim_hTs_max);
        for (int i = 0; i < model->dim_hTs.size(); ++i) {
            const int d = model->dim_hTs[i];
            const int idx = dim_hTs_top[i];
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
        // Eigen::MatrixXd I_cT = Eigen::VectorXd::Ones(model->dim_cT).asDiagonal();
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
    if (model->dim_ecT) {
        Eigen::Ref<const Eigen::MatrixXd> QzxT = ecTx_all;

        Eigen::VectorXd rpT = ECT + RT;
        Eigen::VectorXd rdT = ZT + param.lambdaT + (param.rho * RT);
        
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
        // Eigen::MatrixXd I_ecT = Eigen::VectorXd::Ones(model->dim_ecT).asDiagonal();
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

    for (int t = model->N - 1; t >= 0; --t) {        
        int t_dim_x = t * model->dim_rn;
        int t_dim_u = t * model->dim_u;

        Eigen::Ref<const Eigen::MatrixXd> fx = fx_all.middleCols(t_dim_x, model->dim_rn);
        Eigen::Ref<const Eigen::MatrixXd> fu = fu_all.middleCols(t_dim_u, model->dim_u);

        Eigen::Ref<const Eigen::MatrixXd> qx = qx_all.col(t);
        Eigen::Ref<const Eigen::MatrixXd> qu = qu_all.col(t);

        Eigen::Ref<const Eigen::MatrixXd> qdd = qdd_all.middleCols(t*(model->dim_rn+model->dim_u), model->dim_rn+model->dim_u);
        Eigen::Ref<const Eigen::MatrixXd> qxx = qdd.topLeftCorner(model->dim_rn, model->dim_rn);
        Eigen::Ref<const Eigen::MatrixXd> qxu = qdd.block(0, model->dim_rn, model->dim_rn, model->dim_u);
        Eigen::Ref<const Eigen::MatrixXd> quu = qdd.bottomRightCorner(model->dim_u, model->dim_u);

        Qx = qx + (fx.transpose() * Vx);
        Qu = qu + (fu.transpose() * Vx);
        
        // DDP (TODO: Vector-Hessian Product)
        // vectorHessian(fxx, model->f, fs, x, u, "xx");
        // vectorHessian(fxu, model->f, fs, x, u, "xu");
        // vectorHessian(fuu, model->f, fs, x, u, "uu");
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

        Eigen::Ref<Eigen::VectorXd> ku_ = ku.col(t);
        Eigen::Ref<Eigen::MatrixXd> Ku_ = Ku.middleCols(t_dim_x, model->dim_rn);

        if (model->dim_c) {
            Eigen::Ref<Eigen::VectorXd> s = S.col(t);
            Eigen::Ref<Eigen::VectorXd> y = Y.col(t);
            Eigen::Ref<Eigen::VectorXd> c_v = C.col(t);
            
            Eigen::Ref<Eigen::MatrixXd> Qyx = cx_all.middleCols(t_dim_x, model->dim_rn);
            Eigen::Ref<Eigen::MatrixXd> Qyu = cu_all.middleCols(t_dim_u, model->dim_u);

            Qx += Qyx.transpose() * y;
            Qu += Qyu.transpose() * y;
            
            rp = c_v + s;
            rd.head(model->dim_g) = s.head(model->dim_g).cwiseProduct(y.head(model->dim_g));
            for (int i = 0; i < model->dim_hs.size(); ++i) {
                const int idx = dim_hs_top[i];
                const int n = model->dim_hs[i];
                L_times_vec(rd.segment(idx, n), y.segment(idx, n), s.segment(idx, n));
            }
            rd -= param.mu * e;

            Eigen::VectorXd r(model->dim_c);
            r.head(model->dim_g) = y.head(model->dim_g).cwiseProduct(rp.head(model->dim_g));
            for (int i = 0; i < model->dim_hs.size(); ++i) {
                const int d = model->dim_hs[i];
                const int idx = dim_hs_top[i];
                L_times_vec(r.segment(idx, d), y.segment(idx, d), rp.segment(idx, d));
            }
            r -= rd;

            Sinv_r.head(model->dim_g) = r.head(model->dim_g).cwiseQuotient(s.head(model->dim_g));
            for (int i = 0; i < model->dim_hs.size(); ++i) {
                const int d = model->dim_hs[i];
                const int idx = dim_hs_top[i];
                L_inv_times_vec(Sinv_r.segment(idx, d), s.segment(idx, d), r.segment(idx, d));
            }

            Eigen::VectorXd Sinv_Y_g = y.head(model->dim_g).cwiseQuotient(s.head(model->dim_g));
            Sinv_Y_Qyx.topRows(model->dim_g) = Qyx.topRows(model->dim_g).array().colwise() * Sinv_Y_g.array();
            Sinv_Y_Qyu.topRows(model->dim_g) = Qyu.topRows(model->dim_g).array().colwise() * Sinv_Y_g.array();
            Eigen::MatrixXd SYinv_h_max(dim_hs_max, dim_hs_max);
            for (int i = 0; i < model->dim_hs.size(); ++i) {
                const int d = model->dim_hs[i];
                const int idx = dim_hs_top[i];
                Eigen::Ref<Eigen::MatrixXd> Sinv_Y_h = SYinv_h_max.topLeftCorner(d, d);
                L_inv_times_arrow(Sinv_Y_h, s.segment(idx, d), y.segment(idx, d));
                Sinv_Y_Qyx.middleRows(idx, d) = Sinv_Y_h * Qyx.middleRows(idx, d);
                Sinv_Y_Qyu.middleRows(idx, d) = Sinv_Y_h * Qyu.middleRows(idx, d);
            }
            
            // Inplace Calculation
            ku_ = - (Qu + (Qyu.transpose() * Sinv_r)); // hat_Qu
            Ku_ = - (Qxu + (Qyx.transpose() * Sinv_Y_Qyu)).transpose(); // hat_Qxu
            hat_Quu = Quu + (Qyu.transpose() * Sinv_Y_Qyu);
        }
        
        // TODO
        // Equality Constraint
        // if (model->dim_ec) {
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
        
        // ku.col(t) = ku_;
        // Ku.middleCols(t_dim_x, model->dim_rn) = Ku_;

        opterror = std::max({Qu.lpNorm<Eigen::Infinity>(), opterror});

        // Inequality Constraint
        if (model->dim_c) {
            Eigen::Ref<Eigen::VectorXd> c_v = C.col(t);

            Eigen::Ref<Eigen::MatrixXd> Qyx = cx_all.middleCols(t_dim_x, model->dim_rn);
            Eigen::Ref<Eigen::MatrixXd> Qyu = cu_all.middleCols(t_dim_u, model->dim_u);

            Eigen::Ref<Eigen::VectorXd> ks_ = ks.col(t);
            Eigen::Ref<Eigen::MatrixXd> Ks_ = Ks.middleCols(t_dim_x, model->dim_rn);
            Eigen::Ref<Eigen::VectorXd> ky_ = ky.col(t);
            Eigen::Ref<Eigen::MatrixXd> Ky_ = Ky.middleCols(t_dim_x, model->dim_rn);

            ks_ = - (rp + Qyu * ku_);
            Ks_ = - (Qyx + Qyu * Ku_);
    
            ky_ = Sinv_r + (Sinv_Y_Qyu * ku_);
            Ky_ = Sinv_Y_Qyx + (Sinv_Y_Qyu * Ku_);

            // CHECK: New Value Decrement
            // dV(0) += ky_.transpose() * c_v;
            // dV(1) += ku_.transpose() * Qyu.transpose() * ky_;

            Vx += (Ky_.transpose() * c_v) + (Qyx.transpose() * ky_) + (Ku_.transpose() * Qyu.transpose() * ky_) + (Ky_.transpose() * Qyu * ku_);
            Vxx += (Qyx.transpose() * Ky_) + (Ky_.transpose() * Qyx) + (Ku_.transpose() * Qyu.transpose() * Ky_) + (Ky_.transpose() * Qyu * Ku_);

            // with slack
            // Eigen::VectorXd Qo = Sinv * rd;
            // Eigen::VectorXd Qy = rp;
            // Eigen::MatrixXd I_c = Eigen::VectorXd::Ones(model->dim_c).asDiagonal();
            // dV(0) += Qo.transpose() * ks_;
            // dV(0) += Qy.transpose() * ky_;
            // dV(1) += ky_.transpose() * Qyu * ku_;
            // dV(1) += ky_.transpose() * I_c * ks_; // Qyy = 0

            // Vx += Ky_.transpose() * Qy + Ku_.transpose() * Qyu.transpose() * ky_ + Ky_.transpose() * Qyu * ku_ + Qyx.transpose() * ky_;
            // Vx += Ks_.transpose() * Qo + Ky_.transpose() * I_c * ks_ + Ks_.transpose() * I_c * ky_;

            // Vxx += Qyx.transpose() * Ky_ + Ky_.transpose() * Qyx + Ky_.transpose() * Qyu * Ku_ + Ku_.transpose() * Qyu.transpose() * Ky_;
            // Vxx += Ky_.transpose() * I_c * Ks_ + Ks_.transpose() * I_c * Ky_;

            // ky.col(t) = ky_;
            // Ky.middleCols(t_dim_x, model->dim_rn) = Ky_;
            // ks.col(t) = ks_;
            // Ks.middleCols(t_dim_x, model->dim_rn) = Ks_;

            opterror = std::max({rp.lpNorm<Eigen::Infinity>(), rd.lpNorm<Eigen::Infinity>(), opterror});
            opterror_rp_c = std::max({rp.lpNorm<Eigen::Infinity>(), opterror_rp_c});
            opterror_rd_c = std::max({rd.lpNorm<Eigen::Infinity>(), opterror_rd_c});
        }

        // TODO
        // Equality Constraint
        // if (model->dim_ec) {

        // }
    }
    // std::cout << "opterror_rpT_ec = " << opterror_rpT_ec << std::endl;
    // std::cout << "opterror_rdT_ec = " << opterror_rdT_ec << std::endl;
    // std::cout << "opterror_rpT_c = " << opterror_rpT_c << std::endl;
    // std::cout << "opterror_rdT_c = " << opterror_rdT_c << std::endl;
    // std::cout << "opterror_rp_c = " << opterror_rp_c << std::endl;
    // std::cout << "opterror_rd_c = " << opterror_rd_c << std::endl;
}

void IPDDP::checkRegulate() {
    if (forward_failed || backward_failed) {++regulate;}
    // else if (step <= 3) {regulate = regulate;}
    // else {--regulate;}

    if (regulate < 0) {regulate = 0;}
    else if (param.max_regularization < regulate) {regulate = param.max_regularization;}
}

Eigen::MatrixXd IPDDP::L(const Eigen::VectorXd& x) {
    Eigen::MatrixXd Lx = (x(0) * Eigen::VectorXd::Ones(x.rows())).asDiagonal();
    Lx.col(0) = x;
    Lx.row(0) = x.transpose();
    return Lx;
}

void IPDDP::L_inv_times_vec(Eigen::Ref<Eigen::VectorXd> out, const Eigen::Ref<const Eigen::VectorXd>& soc, const Eigen::Ref<const Eigen::VectorXd>& vec) {
    const double& s = soc(0);
    const int n = soc.size() - 1;
    const auto& v = soc.tail(n);
    const double denom = s * s - v.squaredNorm();
    
    out(0) = (s * vec(0) - v.dot(vec.tail(n)))/ denom;
    out.tail(n).noalias() = (- out(0) / s) * v;
    out.tail(n) += (vec.tail(n) / s);
}

void IPDDP::L_times_vec(Eigen::Ref<Eigen::VectorXd> out, const Eigen::Ref<const Eigen::VectorXd>& soc, const Eigen::Ref<const Eigen::VectorXd>& vec) {
    const double& s = soc(0);
    const int n = soc.size() - 1;
    const auto& v = soc.tail(n);
    
    out(0) = s * vec(0) + v.dot(vec.tail(n));
    out.tail(n) = vec(0) * v + s * vec.tail(n);
}

void IPDDP::L_inv(Eigen::Ref<Eigen::MatrixXd> out, const Eigen::Ref<const Eigen::VectorXd>& soc) {
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

void IPDDP::L_inv_arrow(Eigen::Ref<Eigen::MatrixXd> out, const Eigen::Ref<const Eigen::MatrixXd>& L_inv, const Eigen::Ref<const Eigen::VectorXd>& soc) {
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

void IPDDP::L_inv_times_arrow(Eigen::Ref<Eigen::MatrixXd> out, const Eigen::Ref<const Eigen::VectorXd>& soc_inv, const Eigen::Ref<const Eigen::VectorXd>& soc_arrow) {
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

void IPDDP::forwardPass() {
    Eigen::VectorXd dx;
    Eigen::MatrixXd X_new(model->dim_x, model->N+1);
    Eigen::MatrixXd U_new(model->dim_u, model->N);
    Eigen::MatrixXd S_new(model->dim_c, model->N);
    Eigen::MatrixXd Y_new(model->dim_c, model->N);
    Eigen::MatrixXd C_new(model->dim_c, model->N);
    Eigen::MatrixXd R_new(model->dim_ec, model->N);
    Eigen::MatrixXd Z_new(model->dim_ec, model->N);
    Eigen::MatrixXd EC_new(model->dim_ec, model->N);

    Eigen::VectorXd dxT;
    Eigen::VectorXd ST_new(model->dim_cT);
    Eigen::VectorXd YT_new(model->dim_cT);
    Eigen::VectorXd CT_new(model->dim_cT);
    Eigen::VectorXd RT_new(model->dim_ecT);
    Eigen::VectorXd ZT_new(model->dim_ecT);
    Eigen::VectorXd ECT_new(model->dim_ecT);

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

        dV_exp = -(step_size * dV(0) + step_size * step_size * dV(1));
        // CHECK: Using Expected Value Decrement -> For Early Termination
        if (param.forward_early_termination) {
            if (error <= param.tolerance && dV_exp > 0) {
                forward_failed = 3; continue;
            }
        }

        X_new.col(0) = X.col(0);
        for (int t = 0; t < model->N; ++t) {
            const int t_dim_x = t * model->dim_rn;
            dx = model->perturb(X_new.col(t), X.col(t));
            U_new.col(t) = U.col(t) + (step_size * ku.col(t)) + Ku.middleCols(t_dim_x, model->dim_rn) * dx;
            X_new.col(t+1) = model->f(X_new.col(t), U_new.col(t)).cast<double>();
            if (model->dim_c) {
                S_new.col(t) = S.col(t) + (step_size * ks.col(t)) + Ks.middleCols(t_dim_x, model->dim_rn) * dx;
                Y_new.col(t) = Y.col(t) + (step_size * ky.col(t)) + Ky.middleCols(t_dim_x, model->dim_rn) * dx;
                if (model->dim_g) {
                    const int d = model->dim_g;
                    const Eigen::Ref<const Eigen::VectorXd> S_new_head = S_new.col(t).head(d);
                    const Eigen::Ref<const Eigen::VectorXd> Y_new_head = Y_new.col(t).head(d);
                    const Eigen::Ref<const Eigen::VectorXd> S_head = S.col(t).head(d);
                    const Eigen::Ref<const Eigen::VectorXd> Y_head = Y.col(t).head(d);
                    if ((S_new_head.array() < (one_tau) * S_head.array()).any() || (Y_new_head.array() < (one_tau) * Y_head.array()).any()) {
                        forward_failed = 11; break;
                    }
                }
                for (int i = 0; i < model->dim_hs.size(); ++i) {
                    const int n = model->dim_hs[i] - 1;
                    const int idx = dim_hs_top[i];
                    const double S_new_norm = S_new(idx, t) - S_new.col(t).segment(idx+1, n).norm();
                    const double Y_new_norm = Y_new(idx, t) - Y_new.col(t).segment(idx+1, n).norm();
                    const double S_norm = (one_tau) * (S(idx, t) - S.col(t).segment(idx+1, n).norm());
                    const double Y_norm = (one_tau) * (Y(idx, t) - Y.col(t).segment(idx+1, n).norm());
                    if (S_new_norm < S_norm || Y_new_norm < Y_norm) {
                        forward_failed = 13; break;
                    }
                }
                if (forward_failed) {break;}
                
                C_new.col(t) = model->c(X_new.col(t), U_new.col(t)).cast<double>();
            }
            if (model->dim_ec) {
                R_new.col(t) = R.col(t) + (step_size * kr.col(t)) + Kr.middleCols(t_dim_x, model->dim_rn) * dx;
                Z_new.col(t) = Z.col(t) + (step_size * kz.col(t)) + Kz.middleCols(t_dim_x, model->dim_rn) * dx;

                EC_new.col(t) = model->ec(X_new.col(t), U_new.col(t)).cast<double>();
            }        
        }
        if (forward_failed) {continue;}
        
        dxT = model->perturb(X_new.col(model->N), X.col(model->N));
        if (model->dim_cT) {
            ST_new = ST + (step_size * ksT) + KsT * dxT;
            YT_new = YT + (step_size * kyT) + KyT * dxT;
            if (model->dim_gT) {
                const int d = model->dim_gT;
                const Eigen::Ref<const Eigen::VectorXd> ST_new_head = ST_new.head(d);
                const Eigen::Ref<const Eigen::VectorXd> YT_new_head = YT_new.head(d);
                const Eigen::Ref<const Eigen::VectorXd> ST_head = ST.head(d);
                const Eigen::Ref<const Eigen::VectorXd> YT_head = YT.head(d);
                if ((ST_new_head.array() < (one_tau) * ST_head.array()).any() || (YT_new_head.array() < (one_tau) * YT_head.array()).any()) {
                    forward_failed = 21; continue;
                }
            }
            for (int i = 0; i < model->dim_hTs.size(); ++i) {
                const int n = model->dim_hTs[i] - 1;
                const int idx = dim_hTs_top[i];
                const double ST_new_norm = ST_new(idx) - ST_new.segment(idx+1, n).norm();
                const double YT_new_norm = YT_new(idx) - YT_new.segment(idx+1, n).norm();
                const double ST_norm = (one_tau) * (ST(idx) - ST.segment(idx+1, n).norm());
                const double YT_norm = (one_tau) * (YT(idx) - YT.segment(idx+1, n).norm());
                if (ST_new_norm < ST_norm || YT_new_norm < YT_norm) {
                    forward_failed = 23; break;
                }
            }
            if (forward_failed) {continue;}

            CT_new = model->cT(X_new.col(model->N)).cast<double>();
        }

        if (model->dim_ecT) {
            RT_new = RT + (step_size * krT) + KrT * dxT;
            ZT_new = ZT + (step_size * kzT) + KzT * dxT;

            ECT_new = model->ecT(X_new.col(model->N)).cast<double>();
        }
        
        error_new = 0.0;
        if (model->dim_ec) {error_new += (EC_new + R_new).array().abs().sum();}
        if (model->dim_c) {error_new += (C_new + S_new).array().abs().sum();}
        if (model->dim_ecT) {error_new += (ECT_new + RT_new).array().abs().sum();}
        if (model->dim_cT) {error_new += (CT_new + ST_new).array().abs().sum();}
        // param.tolerance = std::min(param.tolerance, 1.0 / param.rho);
        error_new = std::max(param.tolerance, error_new);

        // Cost
        barriercost_new = 0.0;
        barriercostT_new = 0.0;
        alcost_new = 0.0;
        alcostT_new = 0.0;
        cost_new = calculateTotalCost(X_new, U_new);
        if (model->dim_g) {barriercost_new += S_new.topRows(model->dim_g).array().log().sum();}
        for (int i = 0; i < model->dim_hs.size(); ++i) {barriercost_new += log(S_new.row(dim_hs_top[i]).array().pow(2.0).sum() - S_new.middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).array().pow(2.0).sum())/2;}
        if (model->dim_gT) {barriercostT_new += ST_new.topRows(model->dim_gT).array().log().sum();}
        for (int i = 0; i < model->dim_hTs.size(); ++i) {barriercostT_new += log(ST_new.row(dim_hTs_top[i]).array().pow(2.0).sum() - ST_new.middleRows(dim_hTs_top[i]+1, model->dim_hTs[i]-1).array().pow(2.0).sum())/2;}

        if (model->dim_ec) {alcost_new += (param.lambda.transpose() * R_new).sum() + (0.5 * param.rho * R_new.squaredNorm());}
        if (model->dim_ecT) {alcostT_new += (param.lambdaT.transpose() * RT_new) + (0.5 * param.rho * RT_new.squaredNorm());}
        logcost_new = cost_new - (param.mu * barriercost_new + param.muT * barriercostT_new) + (alcost_new + alcostT_new);
        if (isnan(logcost_new)) {forward_failed = 5; continue;}
        dV_act = logcost - logcost_new;
        
        // Fixed Dual Variable
        // if (model->dim_ec) {for (int t = 0; t < model->N; ++t) {logcost_new += Z_new.col(t).transpose() * (EC_new.col(t) + R_new.col(t));}}
        // if (model->dim_c) {for (int t = 0; t < model->N; ++t) {logcost_new += Y_new.col(t).transpose() * (C_new.col(t) + S_new.col(t));}}
        // if (model->dim_ecT) {logcost_new += ZT_new.transpose() * (ECT_new + RT_new);}
        // if (model->dim_cT) {logcost_new += YT_new.transpose() * (CT_new + ST_new);}
        
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
        X = X_new;
        U = U_new;
        if (model->dim_c) {
            S = S_new;
            Y = Y_new;
            C = C_new;
        }
        if (model->dim_ec) {
            R = R_new;
            Z = Z_new;
            EC = EC_new;
        }
        if (model->dim_cT) {
            ST = ST_new;
            YT = YT_new;
            CT = CT_new;
        }
        if (model->dim_ecT) {
            RT = RT_new;
            ZT = ZT_new;
            ECT = ECT_new;
        }
    }
    // else {std::cout<<"Forward Failed"<<std::endl;}
}

Eigen::MatrixXd IPDDP::getResX() {
    return X;
}

Eigen::MatrixXd IPDDP::getResU() {
    return U;
}

std::vector<double> IPDDP::getAllCost() {
    return all_cost;
}

void IPDDP::logPrint() {
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