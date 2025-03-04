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

    Eigen::MatrixXd X; // State
    Eigen::MatrixXd U; // Input
    
    Eigen::MatrixXd Z; // Equality Lagrangian Multiplier
    Eigen::MatrixXd R; // Equality Slack
    Eigen::MatrixXd S; // Inequality Lagrangian Multiplier
    Eigen::MatrixXd Y; // Inequality Slack
    Eigen::MatrixXd C; // Inequality Constraint
    Eigen::MatrixXd EC; // Equality Constraint

    Eigen::VectorXd ZT; // Equality Lagrangian Multiplier (Terminal)
    Eigen::VectorXd RT; // Equality Slack (Terminal)
    Eigen::VectorXd ST; // Inequality Lagrangian Multiplier (Terminal)
    Eigen::VectorXd YT; // Inequality Slack (Terminal)
    Eigen::VectorXd CT; // Inequality Constraint (Terminal)
    Eigen::VectorXd ECT; // Equality Constraint (Terminal)

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
    Eigen::MatrixXd ky; // Inequality Slack Feedforward Gain
    Eigen::MatrixXd ks; // Inequality Lagrangian Multiplier Feedforward Gain
    
    Eigen::MatrixXd Ku; // Input Feedback Gain
    Eigen::MatrixXd Kr; // Equality Slack Feedback Gain
    Eigen::MatrixXd Kz; // Equality Lagrangian Multiplier Feedback Gain
    Eigen::MatrixXd Ky; // Inequality Slack Feedback Gain
    Eigen::MatrixXd Ks; // Inequality Lagrangian Multiplier Feedback Gain

    Eigen::VectorXd krT; // Equality Slack Feedforward Gain (Terminal)
    Eigen::VectorXd kzT; // Equality Lagrangian Multiplier Feedforward Gain (Terminal)
    Eigen::MatrixXd KrT; // Equality Slack Feedback Gain (Terminal)
    Eigen::MatrixXd KzT; // Equality Lagrangian Multiplier Feedback Gain (Terminal)

    Eigen::VectorXd kyT; // Inequality Slack Feedforward Gain (Terminal)
    Eigen::VectorXd ksT; // Inequality Lagrangian Multiplier Feedforward Gain (Terminal)
    Eigen::MatrixXd KyT; // Inequality Slack Feedback Gain (Terminal)
    Eigen::MatrixXd KsT; // Inequality Lagrangian Multiplier Feedback Gain (Terminal)

    Eigen::VectorXd e;
    Eigen::VectorXd eT;

    double opterror;
    Eigen::VectorXd dV; // Expected Value Change

    std::vector<double> all_cost;

    // Algorithm
    Eigen::MatrixXd L(const Eigen::VectorXd& x);
    void calculateAllDiff();
    void backwardPass();
    void checkRegulate();
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
    else {Z = Eigen::MatrixXd::Ones(model->dim_ec, model->N);}


    if (model->Y_init.size()) {Y = model->Y_init;}
    else {
        Y = Eigen::MatrixXd::Zero(model->dim_c, model->N);
        if (model->dim_g) {Y.topRows(model->dim_g) = Eigen::MatrixXd::Ones(model->dim_g, model->N);}
        for (auto dim_h_top : dim_hs_top) {Y.row(dim_h_top) = Eigen::VectorXd::Ones(model->N);}
    }
    if (model->S_init.size()) {S = model->S_init;}
    else {
        S = Eigen::MatrixXd::Zero(model->dim_c, model->N);
        if (model->dim_g) {S.topRows(model->dim_g) = Eigen::MatrixXd::Ones(model->dim_g, model->N);}
        for (auto dim_h_top : dim_hs_top) {S.row(dim_h_top) = Eigen::VectorXd::Ones(model->N);}
    }

    if (model->RT_init.size()) {RT = model->RT_init;}
    else {RT = Eigen::VectorXd::Zero(model->dim_ecT);}
    if (model->ZT_init.size()) {ZT = model->ZT_init;}
    else {ZT = Eigen::VectorXd::Ones(model->dim_ecT);}

    if (model->YT_init.size()) {YT = model->YT_init;}
    else {
        YT = Eigen::VectorXd::Zero(model->dim_cT);
        if (model->dim_gT) {YT.topRows(model->dim_gT) = Eigen::VectorXd::Ones(model->dim_gT);}
        for (auto dim_hT_top : dim_hTs_top) {YT(dim_hT_top) = 1.0;}
    }
    if (model->ST_init.size()) {ST = model->ST_init;}
    else {
        ST = Eigen::VectorXd::Zero(model->dim_cT);
        if (model->dim_gT) {ST.topRows(model->dim_g) = Eigen::VectorXd::Ones(model->dim_gT);}
        for (auto dim_hT_top : dim_hTs_top) {ST(dim_hT_top) = 1.0;}
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
    }
    if (model->dim_ec) {
        ecx_all.resize(model->dim_ec, model->dim_rn * model->N);
        ecu_all.resize(model->dim_ec, model->dim_u * model->N);
    }
    if (model->dim_cT) {cTx_all.resize(model->dim_cT, model->dim_rn);}
    if (model->dim_ecT) {ecTx_all.resize(model->dim_ecT, model->dim_rn);}

    ku.resize(model->dim_u, model->N);
    kr.resize(model->dim_ec, model->N);
    kz.resize(model->dim_ec, model->N);
    ky.resize(model->dim_c, model->N);
    ks.resize(model->dim_c, model->N);
    Ku.resize(model->dim_u, model->dim_rn * model->N);
    Kr.resize(model->dim_ec, model->dim_rn * model->N);
    Kz.resize(model->dim_ec, model->dim_rn * model->N);
    Ky.resize(model->dim_c, model->dim_rn * model->N);
    Ks.resize(model->dim_c, model->dim_rn * model->N);

    krT.resize(model->dim_ecT);
    kzT.resize(model->dim_ecT);
    kyT.resize(model->dim_cT);
    ksT.resize(model->dim_cT);
    KrT.resize(model->dim_ecT, model->dim_rn);
    KzT.resize(model->dim_ecT, model->dim_rn);
    KyT.resize(model->dim_cT, model->dim_rn);
    KsT.resize(model->dim_cT, model->dim_rn);

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
    resetFilter();
    resetRegulation();

    for (int i = 0; i <= this->param.max_step_iter; ++i) {
        step_list.push_back(std::pow(2.0, static_cast<double>(-i)));
    }
}

void IPDDP::initialRoll() {
    for (int t = 0; t < model->N; ++t) {
        X.col(t+1) = model->f(X.col(t), U.col(t)).cast<double>();
    }
    if (model->dim_ec) {
        EC.resize(model->dim_ec, model->N);
        for (int t = 0; t < model->N; ++t) {
            EC.col(t) = model->ec(X.col(t), U.col(t)).cast<double>();
        }
    }
    if (model->dim_c) {
        C.resize(model->dim_c, model->N);
        for (int t = 0; t < model->N; ++t) {
            C.col(t) = model->c(X.col(t), U.col(t)).cast<double>();
        }
    }
    if (model->dim_cT) {CT = model->cT(X.col(model->N)).cast<double>();}
    if (model->dim_ecT) {ECT = model->ecT(X.col(model->N)).cast<double>();}

    if (model->dim_ec) {R = -EC;}
    if (model->dim_ecT) {RT = -ECT;}

    cost = calculateTotalCost(X, U);
}

void IPDDP::resetFilter() {
    double barriercost = 0.0;
    double barriercostT = 0.0;
    double alcost = 0.0;
    double alcostT = 0.0;

    if (model->dim_g) {barriercost += Y.topRows(model->dim_g).array().log().sum();}
    for (int i = 0; i < model->dim_hs.size(); ++i) {barriercost += log(Y.row(dim_hs_top[i]).array().pow(2.0).sum() - Y.middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).array().pow(2.0).sum())/2;}
    if (model->dim_gT) {barriercostT += YT.topRows(model->dim_gT).array().log().sum();}
    for (int i = 0; i < model->dim_hTs.size(); ++i) {barriercostT += log(YT.row(dim_hTs_top[i]).array().pow(2.0).sum() - YT.middleRows(dim_hTs_top[i]+1, model->dim_hTs[i]-1).array().pow(2.0).sum())/2;}

    // CHECK
    if (model->dim_ec) {alcost += (param.lambda.transpose() * R).sum() + (0.5 * param.rho * R.squaredNorm());}
    if (model->dim_ecT) {alcostT += (param.lambdaT.transpose() * RT) + (0.5 * param.rho * RT.squaredNorm());}

    logcost = cost - param.mu * (barriercost + barriercostT) + (alcost + alcostT);

    error = 0.0;
    if (model->dim_ec) {error += (EC + R).colwise().lpNorm<1>().sum();}
    if (model->dim_c) {error += (C + Y).colwise().lpNorm<1>().sum();}
    if (model->dim_ecT) {error += (ECT + RT).lpNorm<1>();}
    if (model->dim_cT) {error += (CT + YT).lpNorm<1>();}
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
    << std::setw(16) << "LogCost"
    << std::setw(22) << "OptError"
    << std::setw(13) << "Error"
    << std::setw(4) << "Reg"
    << std::setw(7) << "Step" << std::endl;

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
            }
            
            this->logPrint();
            
            all_cost.push_back(cost);
    
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
        }
        if (param.max_iter < iter) {
            std::cout << "Max Iteration" << std::endl;
            return;
        }

        // CHECK
        if ((param.mu <= param.mu_min) && (param.rho_max <= param.rho)) {
            std::cout << "Outer Max/Min" << std::endl;
            return;
        }

        // Update Outer Loop Parameters
        param.mu = std::max((param.mu_min), std::min(param.mu_mul * param.mu, std::pow(param.mu, param.mu_exp)));
        param.lambdaT = param.lambdaT + param.rho * RT;
        // CHECK
        // param.lambdaT = 2 * (param.lambdaT + param.rho * RT) - ZT;
        param.rho = std::min(param.rho_max, std::max(param.rho_mul * param.rho, 1.0 / param.mu));
        resetFilter();
        resetRegulation();
    }
}

Eigen::MatrixXd IPDDP::L(const Eigen::VectorXd& x) {
    Eigen::MatrixXd Lx = (x(0) * Eigen::VectorXd::Ones(x.rows())).asDiagonal();
    Lx.col(0) = x;
    Lx.row(0) = x.transpose();
    return Lx;
}

void IPDDP::calculateAllDiff() {
    // CHECK: Multithreading (TODO: with CUDA)
    // CHECK 1: Making branch in for loop is fine for parallelization?
    // CHECK 2: Move to Model to make solver only consider Eigen (not autodiff::dual)

    // #pragma omp parallel for
    for (int t = 0; t < model->N; ++t) {
        VectorXdual2nd x = X.col(t).cast<dual2nd>();
        VectorXdual2nd u = U.col(t).cast<dual2nd>();

        fx_all.middleCols(t*model->dim_rn, model->dim_rn) = model->fx(x,u);
        fu_all.middleCols(t*model->dim_u, model->dim_u) = model->fu(x,u);
        qx_all.col(t) = model->qx(x,u);
        qu_all.col(t) = model->qu(x,u);
        qdd_all.middleCols(t*(model->dim_rn+model->dim_u), model->dim_rn+model->dim_u) = model->qdd(x,u);
        // if (DDP) {
        //     fxx_all.middleCols(t*model->dim_rn*model->dim_rn, model->dim_rn*model->dim_rn) = model->fxx(x,u);
        //     fxu_all.middleCols(t*model->dim_rn*model->dim_u, model->dim_rn*model->dim_u) = model->fxu(x,u);
        //     fuu_all.middleCols(t*model->dim_u*model->dim_u, model->dim_u*model->dim_u) = model->fuu(x,u);
        // }
        if (model->dim_c) {
            cx_all.middleCols(t*model->dim_rn, model->dim_rn) = model->cx(x,u);
            cu_all.middleCols(t*model->dim_u, model->dim_u) = model->cu(x,u);
        }
        if (model->dim_ec) {
            ecx_all.middleCols(t*model->dim_rn, model->dim_rn) = model->ecx(x,u);
            ecu_all.middleCols(t*model->dim_u, model->dim_u) = model->ecu(x,u);
        }
    }
    VectorXdual2nd xT = X.col(model->N).cast<dual2nd>();
    px_all = model->px(xT);
    pxx_all = model->pxx(xT);
    if (model->dim_cT) {cTx_all = model->cTx(xT);}
    if (model->dim_ecT) {ecTx_all = model->ecTx(xT);}
}

void IPDDP::backwardPass() {
    Eigen::VectorXd y(model->dim_c);
    Eigen::VectorXd s(model->dim_c);
    Eigen::VectorXd c_v(model->dim_c);

    Eigen::MatrixXd Y_;
    Eigen::MatrixXd S_;

    Eigen::VectorXd Vx(model->dim_rn);
    Eigen::MatrixXd Vxx(model->dim_rn,model->dim_rn);

    Eigen::MatrixXd fx(model->dim_rn,model->dim_rn), fu(model->dim_rn,model->dim_u);
    Eigen::MatrixXd Qsx(model->dim_c,model->dim_rn), Qsu(model->dim_c,model->dim_u);
    Eigen::MatrixXd Qzx(model->dim_ec,model->dim_rn), Qzu(model->dim_ec,model->dim_u);

    // Eigen::Tensor<double, 3> fxx(model->dim_rn,model->dim_rn,model->dim_rn);
    // Eigen::Tensor<double, 3> fxu(model->dim_rn,model->dim_rn,model->dim_u);
    // Eigen::Tensor<double, 3> fuu(model->dim_rn,model->dim_u,model->dim_u);

    Eigen::VectorXd qx(model->dim_rn), qu(model->dim_u);
    Eigen::MatrixXd qdd(model->dim_rn+model->dim_u, model->dim_rn+model->dim_u);
    Eigen::MatrixXd qxx(model->dim_rn,model->dim_rn), qxu(model->dim_rn,model->dim_u), quu(model->dim_u,model->dim_u);

    Eigen::VectorXd Qx(model->dim_rn), Qu(model->dim_u);
    Eigen::MatrixXd Qxx(model->dim_rn,model->dim_rn), Qxu(model->dim_rn,model->dim_u), Quu(model->dim_u,model->dim_u);
    Eigen::MatrixXd Qxu_reg(model->dim_rn,model->dim_u), Quu_reg(model->dim_u,model->dim_u);
    Eigen::MatrixXd Quu_sim(model->dim_u,model->dim_u);

    Eigen::MatrixXd Yinv;
    Eigen::MatrixXd SYinv;

    Eigen::VectorXd rp;
    Eigen::VectorXd rd;
    Eigen::VectorXd r;

    Eigen::LLT<Eigen::MatrixXd> Quu_llt;
    Eigen::MatrixXd R;

    Eigen::VectorXd ku_(model->dim_u);
    Eigen::VectorXd kr_(model->dim_ec);
    Eigen::VectorXd kz_(model->dim_ec);
    Eigen::VectorXd ky_(model->dim_c);
    Eigen::VectorXd ks_(model->dim_c);
    Eigen::MatrixXd Ku_(model->dim_u, model->dim_rn);
    Eigen::MatrixXd Kr_(model->dim_ec, model->dim_rn);
    Eigen::MatrixXd Kz_(model->dim_ec, model->dim_rn);
    Eigen::MatrixXd Ky_(model->dim_c, model->dim_rn);
    Eigen::MatrixXd Ks_(model->dim_c, model->dim_rn);

    opterror = 0.0;

    dV = Eigen::VectorXd::Zero(2);

    checkRegulate();

    double reg1_mu = param.reg1_min * (std::pow(param.reg1_exp, regulate));
    double reg2_mu = param.reg2_min * (std::pow(param.reg2_exp, regulate));

    Vx = px_all;
    Vxx = pxx_all;

    // Inequality Terminal Constraint
    if (model->dim_cT) {
        Eigen::MatrixXd YT_ = Eigen::MatrixXd::Zero(model->dim_cT, model->dim_cT);
        Eigen::MatrixXd ST_ = Eigen::MatrixXd::Zero(model->dim_cT, model->dim_cT);
        if (model->dim_gT) {
            YT_.topLeftCorner(model->dim_gT, model->dim_gT) = YT.topRows(model->dim_gT).asDiagonal();
            ST_.topLeftCorner(model->dim_gT, model->dim_gT) = ST.topRows(model->dim_gT).asDiagonal();
        }
        for (int i = 0; i < model->dim_hTs.size(); ++i) {
            YT_.block(dim_hTs_top[i], dim_hTs_top[i], model->dim_hTs[i], model->dim_hTs[i]) = L(YT.middleRows(dim_hTs_top[i], model->dim_hTs[i]));
            ST_.block(dim_hTs_top[i], dim_hTs_top[i], model->dim_hTs[i], model->dim_hTs[i]) = L(ST.middleRows(dim_hTs_top[i], model->dim_hTs[i]));
        }
        Eigen::MatrixXd YTinv = YT_.inverse();
        Eigen::MatrixXd STYTinv = YTinv * ST_;

        Eigen::MatrixXd QsxT = cTx_all;
        
        Eigen::VectorXd rpT = CT + YT;
        Eigen::VectorXd rdT = YT_*ST - param.mu*eT;
        Eigen::VectorXd rT = ST_*rpT - rdT;
        
        kyT = - rpT;
        KyT = - QsxT;
        ksT = YTinv * rT;
        KsT = STYTinv * QsxT;

        // CHECK: New Value Decrement
        dV(0) += ksT.transpose() * rpT;

        Vx += KsT.transpose() * CT + QsxT.transpose() * ksT;
        Vxx += QsxT.transpose() * KsT + KsT.transpose() * QsxT;

        opterror = std::max({rpT.lpNorm<Eigen::Infinity>(), rdT.lpNorm<Eigen::Infinity>(), opterror});
    }

    // Equality Terminal Constraint
    if (model->dim_ecT) {
        Eigen::MatrixXd QzxT = ecTx_all;

        Eigen::VectorXd rpT = ECT + RT;
        Eigen::VectorXd rdT = ZT + param.lambdaT + (param.rho * RT);
        Eigen::VectorXd rT = param.rho*rpT - rdT;

        krT = - rpT;
        KrT = - QzxT;
        kzT = rT;
        KzT = param.rho * QzxT;

        // CHECK: New Value Decrement
        dV(0) += kzT.transpose() * rpT;

        Vx += KzT.transpose() * ECT + QzxT.transpose() * kzT;
        Vxx += QzxT.transpose() * KzT + KzT.transpose() * QzxT;
        
        opterror = std::max({rpT.lpNorm<Eigen::Infinity>(), rdT.lpNorm<Eigen::Infinity>(), opterror});
    }


    backward_failed = false;

    for (int t = model->N - 1; t >= 0; --t) {
        int t_dim_x = t * model->dim_rn;
        int t_dim_u = t * model->dim_u;

        fx = fx_all.middleCols(t_dim_x, model->dim_rn);
        fu = fu_all.middleCols(t_dim_u, model->dim_u);

        qx = qx_all.col(t);
        qu = qu_all.col(t);

        qdd = qdd_all.middleCols(t*(model->dim_rn+model->dim_u), model->dim_rn+model->dim_u);
        qxx = qdd.topLeftCorner(model->dim_rn, model->dim_rn);
        qxu = qdd.block(0, model->dim_rn, model->dim_rn, model->dim_u);
        quu = qdd.bottomRightCorner(model->dim_u, model->dim_u);

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
        Qxx = qxx + (fx.transpose() * Vxx * fx);
        Qxu = qxu + (fx.transpose() * Vxx * fu);
        Quu = quu + (fu.transpose() * Vxx * fu);

        // Inequality Constraint
        if (model->dim_c) {
            y = Y.col(t);
            s = S.col(t);
            c_v = C.col(t);
            
            Y_ = Eigen::MatrixXd::Zero(model->dim_c, model->dim_c);
            S_ = Eigen::MatrixXd::Zero(model->dim_c, model->dim_c);
            if (model->dim_g) {
                Y_.topLeftCorner(model->dim_g, model->dim_g) = y.topRows(model->dim_g).asDiagonal();
                S_.topLeftCorner(model->dim_g, model->dim_g) = s.topRows(model->dim_g).asDiagonal();
            }
            for (int i = 0; i < model->dim_hs.size(); ++i) {
                Y_.block(dim_hs_top[i], dim_hs_top[i], model->dim_hs[i], model->dim_hs[i]) = L(y.middleRows(dim_hs_top[i], model->dim_hs[i]));
                S_.block(dim_hs_top[i], dim_hs_top[i], model->dim_hs[i], model->dim_hs[i]) = L(s.middleRows(dim_hs_top[i], model->dim_hs[i]));
            }
            Yinv = Y_.inverse();
            SYinv = Yinv * S_;
            
            Qsx = cx_all.middleCols(t_dim_x, model->dim_rn);
            Qsu = cu_all.middleCols(t_dim_u, model->dim_u);
            
            rp = c_v + y;
            rd = Y_*s - param.mu*e;
            r = S_*rp - rd;
            
            Qx += Qsx.transpose() * s;
            Qu += Qsu.transpose() * s;

            // CHECK (Orignal IPDDP for Value Update)
            // Qx += Qsx.transpose() * s + Qsx.transpose() * (Yinv * r);
            // Qu += Qsu.transpose() * s + Qsu.transpose() * (Yinv * r);
            
            // Qxx += Qsx.transpose() * SYinv * Qsx;
            // Qxu += Qsx.transpose() * SYinv * Qsu;
            // Quu += Qsu.transpose() * SYinv * Qsu;
        }
        
        // Regularization
        Qxu_reg = Qxu + fx.transpose() * (reg1_mu * Eigen::MatrixXd::Identity(model->dim_rn, model->dim_rn)) * fu;
        Quu_reg = Quu + fu.transpose() * (reg1_mu * Eigen::MatrixXd::Identity(model->dim_rn, model->dim_rn)) * fu;
        Quu_reg += reg2_mu * Eigen::MatrixXd::Identity(model->dim_u, model->dim_u);
        
        // TODO
        // Equality Constraint
        // if (model->dim_ec) {

        // }

        Quu_sim = 0.5*(Quu_reg + Quu_reg.transpose());
        Quu_reg = Quu_sim;
        Quu_llt = Eigen::LLT<Eigen::MatrixXd>(Quu_reg);
        if (!Quu_reg.isApprox(Quu_reg.transpose()) || Quu_llt.info() == Eigen::NumericalIssue) {
            backward_failed = true;
            break;
        }
        R = Quu_llt.matrixU();

        ku_ = -R.inverse() * (R.transpose().inverse() * Qu);
        Ku_ = -R.inverse() * (R.transpose().inverse() * Qxu_reg.transpose());
        
        dV(0) += ku_.transpose() * Qu;
        dV(1) += 0.5 * ku_.transpose() * Quu_reg * ku_;
        
        Vx = Qx + (Ku_.transpose() * Qu) + (Ku_.transpose() * Quu * ku_) + (Qxu * ku_);
        Vxx = Qxx + (Ku_.transpose() * Qxu.transpose()) + (Qxu * Ku_) + (Ku_.transpose() * Quu * Ku_);
        
        ku.col(t) = ku_;
        Ku.middleCols(t_dim_x, model->dim_rn) = Ku_;

        opterror = std::max({Qu.lpNorm<Eigen::Infinity>(), opterror});

        // Inequality Constraint
        if (model->dim_c) {
            ks_ = (Yinv * r) + (SYinv * Qsu * ku_);
            Ks_ = SYinv * (Qsx + Qsu * Ku_);
            ky_ = -rp - Qsu * ku_;
            Ky_ = -Qsx - Qsu * Ku_;

            // CHECK: New Value Decrement
            dV(0) += ks_.transpose() * rp;
            dV(1) += ku_.transpose() * Qsu.transpose() * ks_;

            Vx += (Ks_.transpose() * c_v) + (Qsx.transpose() * ks_) + (Ku_.transpose() * Qsu.transpose() * ks_) + (Ks_.transpose() * Qsu * ku_);
            Vxx += (Qsx.transpose() * Ks_) + (Ks_.transpose() * Qsx) + (Ku_.transpose() * Qsu.transpose() * Ks_) + (Ks_.transpose() * Qsu * Ku_);

            ks.col(t) = ks_;
            Ks.middleCols(t_dim_x, model->dim_rn) = Ks_;
            ky.col(t) = ky_;
            Ky.middleCols(t_dim_x, model->dim_rn) = Ky_;

            opterror = std::max({rp.lpNorm<Eigen::Infinity>(), rd.lpNorm<Eigen::Infinity>(), opterror});
        }

        // TODO
        // Equality Constraint
        // if (model->dim_ec) {

        // }
    }
}

void IPDDP::checkRegulate() {
    if (forward_failed || backward_failed) {++regulate;}
    // else if (step == 0) {--regulate;}
    // else if (step <= 3) {regulate = regulate;}
    else {--regulate;}

    if (regulate < 0) {regulate = 0;}
    else if (param.max_regularization < regulate) {regulate = param.max_regularization;}
}

void IPDDP::forwardPass() {
    Eigen::VectorXd dx;
    Eigen::MatrixXd X_new(model->dim_x, model->N+1);
    Eigen::MatrixXd U_new(model->dim_u, model->N);
    Eigen::MatrixXd Y_new(model->dim_c, model->N);
    Eigen::MatrixXd S_new(model->dim_c, model->N);
    Eigen::MatrixXd C_new(model->dim_c, model->N);
    Eigen::MatrixXd R_new(model->dim_ec, model->N);
    Eigen::MatrixXd Z_new(model->dim_ec, model->N);
    Eigen::MatrixXd EC_new(model->dim_ec, model->N);

    Eigen::VectorXd dxT;
    Eigen::VectorXd YT_new(model->dim_cT);
    Eigen::VectorXd ST_new(model->dim_cT);
    Eigen::VectorXd CT_new(model->dim_cT);
    Eigen::VectorXd RT_new(model->dim_ecT);
    Eigen::VectorXd ZT_new(model->dim_ecT);
    Eigen::VectorXd ECT_new(model->dim_ecT);

    double tau = std::max(0.99, 1.0 - param.mu);
    
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

        forward_failed = false;
        double step_size = step_list[step];

        X_new.col(0) = X.col(0);
        for (int t = 0; t < model->N; ++t) {
            int t_dim_x = t * model->dim_rn;
            dx = model->perturb(X_new.col(t), X.col(t));
            U_new.col(t) = U.col(t) + (step_size * ku.col(t)) + Ku.middleCols(t_dim_x, model->dim_rn) * dx;
            X_new.col(t+1) = model->f(X_new.col(t), U_new.col(t)).cast<double>();
        }
        if (model->dim_c) {
            for (int t = 0; t < model->N; ++t) {
                int t_dim_x = t * model->dim_rn;
                dx = model->perturb(X_new.col(t), X.col(t));
                Y_new.col(t) = Y.col(t) + (step_size * ky.col(t)) + Ky.middleCols(t_dim_x, model->dim_rn) * dx;
                S_new.col(t) = S.col(t) + (step_size * ks.col(t)) + Ks.middleCols(t_dim_x, model->dim_rn) * dx;
            }
        }
        if (model->dim_ec) {
            for (int t = 0; t < model->N; ++t) {
                int t_dim_x = t * model->dim_rn;
                dx = model->perturb(X_new.col(t), X.col(t));
                R_new.col(t) = R.col(t) + (step_size * kr.col(t)) + Kr.middleCols(t_dim_x, model->dim_rn) * dx;
                Z_new.col(t) = Z.col(t) + (step_size * kz.col(t)) + Kz.middleCols(t_dim_x, model->dim_rn) * dx;
            }
        }
        dxT = model->perturb(X_new.col(model->N), X.col(model->N));
        if (model->dim_cT) {
            YT_new = YT + (step_size * kyT) + KyT * dxT;
            ST_new = ST + (step_size * ksT) + KsT * dxT;
        }
        if (model->dim_ecT) {
            RT_new = RT + (step_size * krT) + KrT * dxT;
            ZT_new = ZT + (step_size * kzT) + KzT * dxT;
        }

        // Fraction to boundary
        for (int t = 0; t < model->N; ++t) {
            if (model->dim_g) {
                if ((Y_new.col(t).topRows(model->dim_g).array() < (1 - tau) * Y.col(t).topRows(model->dim_g).array()).any()) {forward_failed = true; break;}
                if ((S_new.col(t).topRows(model->dim_g).array() < (1 - tau) * S.col(t).topRows(model->dim_g).array()).any()) {forward_failed = true; break;}
            }
            for (int i = 0; i < model->dim_hs.size(); ++i) {
                if ((Y_new.col(t).row(dim_hs_top[i]).array() - Y_new.col(t).middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).norm()
                < (1 - tau) * (Y.col(t).row(dim_hs_top[i]).array() - Y.col(t).middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).norm())).any()) {forward_failed = true; break;}
                if ((S_new.col(t).row(dim_hs_top[i]).array() - S_new.col(t).middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).norm()
                < (1 - tau) * (S.col(t).row(dim_hs_top[i]).array() - S.col(t).middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).norm())).any()) {forward_failed = true; break;}
            }
            if (forward_failed) {break;}
        }
        if (model->dim_cT) {
            if (model->dim_gT) {
                if ((YT_new.topRows(model->dim_gT).array() < (1 - tau) * YT.topRows(model->dim_gT).array()).any()) {forward_failed = true; break;}
                if ((ST_new.topRows(model->dim_gT).array() < (1 - tau) * ST.topRows(model->dim_gT).array()).any()) {forward_failed = true; break;}
            }
            for (int i = 0; i < model->dim_hTs.size(); ++i) {
                if ((YT_new.row(dim_hTs_top[i]).array() - YT_new.middleRows(dim_hTs_top[i]+1, model->dim_hTs[i]-1).norm()
                < (1 - tau) * (YT.row(dim_hTs_top[i]).array() - YT.middleRows(dim_hTs_top[i]+1, model->dim_hTs[i]-1).norm())).any()) {forward_failed = true; break;}
                if ((ST_new.row(dim_hTs_top[i]).array() - ST_new.middleRows(dim_hTs_top[i]+1, model->dim_hTs[i]-1).norm()
                < (1 - tau) * (ST.row(dim_hTs_top[i]).array() - ST.middleRows(dim_hTs_top[i]+1, model->dim_hTs[i]-1).norm())).any()) {forward_failed = true; break;}
            }
        }
        
        if (forward_failed) {continue;}

        // Cost
        barriercost_new = 0.0;
        barriercostT_new = 0.0;
        alcost_new = 0.0;
        alcostT_new = 0.0;
        cost_new = calculateTotalCost(X_new, U_new);
        if (model->dim_g) {barriercost_new += Y_new.topRows(model->dim_g).array().log().sum();}
        for (int i = 0; i < model->dim_hs.size(); ++i) {barriercost_new += log(Y_new.row(dim_hs_top[i]).array().pow(2.0).sum() - Y_new.middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).array().pow(2.0).sum())/2;}
        if (model->dim_gT) {barriercostT_new += YT_new.topRows(model->dim_gT).array().log().sum();}
        for (int i = 0; i < model->dim_hTs.size(); ++i) {barriercostT_new += log(YT_new.row(dim_hTs_top[i]).array().pow(2.0).sum() - YT_new.middleRows(dim_hTs_top[i]+1, model->dim_hTs[i]-1).array().pow(2.0).sum())/2;}

        if (model->dim_ec) {alcost_new += (param.lambda.transpose() * R_new).sum() + (0.5 * param.rho * R_new.squaredNorm());}
        if (model->dim_ecT) {alcostT_new += (param.lambdaT.transpose() * RT_new) + (0.5 * param.rho * RT_new.squaredNorm());}
        logcost_new = cost_new - param.mu * (barriercost_new + barriercostT_new) + (alcost_new + alcostT_new);
        
        // Error
        if (model->dim_c) {
            for (int t = 0; t < model->N; ++t) {
                C_new.col(t) = model->c(X_new.col(t), U_new.col(t)).cast<double>();
            }
        }
        if (model->dim_ec) {
            for (int t = 0; t < model->N; ++t) {
                EC_new.col(t) = model->ec(X_new.col(t), U_new.col(t)).cast<double>();
            }
        }
        if (model->dim_cT) {CT_new = model->cT(X_new.col(model->N)).cast<double>();}
        if (model->dim_ecT) {ECT_new = model->ecT(X_new.col(model->N)).cast<double>();}
        
        error_new = 0.0;
        if (model->dim_ec) {error_new += (EC_new + R_new).colwise().lpNorm<1>().sum();}
        if (model->dim_c) {error_new += (C_new + Y_new).colwise().lpNorm<1>().sum();}
        if (model->dim_ecT) {error_new += (ECT_new + RT_new).lpNorm<1>();}
        if (model->dim_cT) {error_new += (CT_new + YT_new).lpNorm<1>();}
        error_new = std::max(param.tolerance, error_new);

        // 1. With Expected Value Decrement
        dV_act = logcost - logcost_new;
        dV_exp = step * dV(0) + step * step * dV(1);
        // std::cout << dV(0) << " " << dV(1) << std::endl;
        // std::cout << dV_act << " " << dV_exp << std::endl;
        // std::cout << logcost << " " << logcost_new << std::endl;
        if ((1e-4 * dV_exp < dV_act && dV_act < 10 * dV_exp) && error >= error_new) {break;}
        // 2. Only Value Decrement
        // if (logcost >= logcost_new && error >= error_new) {break;}
        
        forward_failed = true;
    }

    if (!forward_failed) {
        cost = cost_new;
        logcost = logcost_new;
        error = error_new;
        X = X_new;
        U = U_new;
        if (model->dim_c) {
            Y = Y_new;
            S = S_new;
            C = C_new;
        }
        if (model->dim_ec) {
            R = R_new;
            Z = Z_new;
            EC = EC_new;
        }
        if (model->dim_cT) {
            YT = YT_new;
            ST = ST_new;
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
              << std::setw(16) << logcost
              << std::setw(22) << opterror
              << std::setw(13) << error
              << std::setw(4) << regulate
              << std::setw(7) << step_list[step] << std::endl;
}