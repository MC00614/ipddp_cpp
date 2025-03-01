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
    
    Eigen::MatrixXd M; // Equality Lagrangian Multiplier
    Eigen::MatrixXd S; // Inequality Lagrangian Multiplier
    Eigen::MatrixXd Y; // Inequality Slack
    Eigen::MatrixXd C; // Inequality Constraint
    Eigen::MatrixXd EC; // Equality Constraint

    Eigen::VectorXd MT; // Equality Lagrangian Multiplier (Terminal)
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

    int iter;
    void resetRegulation();
    int regulate;
    bool backward_failed;

    Eigen::MatrixXd ku; // Input Feedforward Gain 
    Eigen::MatrixXd ky; // Inequality Slack Feedforward Gain
    Eigen::MatrixXd ks; // Inequality Lagrangian Multiplier Feedforward Gain
    Eigen::MatrixXd Ku; // Input Feedback Gain
    Eigen::MatrixXd Ky; // Inequality Slack Feedback Gain
    Eigen::MatrixXd Ks; // Inequality Lagrangian Multiplier Feedback Gain

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
    void backwardPass();
    void checkRegulate();
    void forwardPass();
    double calculateTotalCost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U);
    void logPrint();
};

template<typename ModelClass>
IPDDP::IPDDP(std::shared_ptr<ModelClass> model_ptr) : model(model_ptr) {
    static_assert(std::is_base_of<ModelBase, ModelClass>::value, "ModelClass must be derived from ModelBase");
    if (std::is_base_of<QuatModelBase, ModelClass>::value) {
        model->dim_rn = model->dim_x - 1;
    }
    else {model->dim_rn = model->dim_x;}
    // if (!model->dim_rn) {model->dim_rn = model->dim_x;}

    // Inequality Constraint Stack (TODO: Move to Model)
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
    // Inequality Constraint Stack (Terminal) (TODO: Move to Model)
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

    // Initialization
    if (model->X_init.size()) {X = model->X_init;}
    else {X = Eigen::MatrixXd::Zero(model->dim_x, model->N+1);}
    if (model->U_init.size()) {U = model->U_init;}
    else {U = Eigen::MatrixXd::Zero(model->dim_u, model->N);}

    if (model->M_init.size()) {M = model->M_init;}
    else {M = Eigen::MatrixXd::Ones(model->dim_ec, model->N);}
    if (model->Y_init.size()) {Y = model->Y_init;}
    else {
        Y = Eigen::MatrixXd::Zero(model->dim_c, model->N);
        if (model->dim_g) {Y.topRows(model->dim_g) = Eigen::MatrixXd::Ones(model->dim_g, model->N);}
        for (auto dim_h_top : dim_hs_top) {Y.row(dim_h_top) = Eigen::VectorXd::Ones(model->N);}
    }
    if (model->S_init.size()) {S = model->S_init;}
    else {
        S = Eigen::MatrixXd::Zero(model->dim_c, model->N);
        if (model->dim_g) {S.topRows(model->dim_g) = 0.01*Eigen::MatrixXd::Ones(model->dim_g, model->N);}
        for (auto dim_h_top : dim_hs_top) {S.row(dim_h_top) = 0.01*Eigen::VectorXd::Ones(model->N);}
    }

    if (model->MT_init.size()) {MT = model->MT_init;}
    else {MT = 0.01*Eigen::VectorXd::Ones(model->dim_ecT);}
    if (model->YT_init.size()) {YT = model->YT_init;}
    else {
        YT = Eigen::VectorXd::Zero(model->dim_cT);
        if (model->dim_gT) {Y.topRows(model->dim_gT) = Eigen::VectorXd::Ones(model->dim_gT);}
        for (auto dim_hT_top : dim_hTs_top) {YT(dim_hT_top) = 1.0;}
    }
    if (model->ST_init.size()) {ST = model->ST_init;}
    else {
        ST = Eigen::VectorXd::Zero(model->dim_cT);
        if (model->dim_gT) {S.topRows(model->dim_g) = 0.01*Eigen::VectorXd::Ones(model->dim_gT);}
        for (auto dim_hT_top : dim_hTs_top) {ST(dim_hT_top) = 0.01;}
    }

    ku.resize(model->dim_u, model->N);
    ky.resize(model->dim_c, model->N);
    ks.resize(model->dim_c, model->N);
    Ku.resize(model->dim_u, model->dim_rn * model->N);
    Ky.resize(model->dim_c, model->dim_rn * model->N);
    Ks.resize(model->dim_c, model->dim_rn * model->N);

    kyT.resize(model->dim_cT);
    ksT.resize(model->dim_cT);
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
    cost = calculateTotalCost(X, U);
}

void IPDDP::resetFilter() {
    double barriercost = 0.0;
    double barriercostTerminal = 0.0;
    if (model->dim_g) {barriercost += Y.topRows(model->dim_g).array().log().sum();}
    for (int i = 0; i < model->dim_hs.size(); ++i) {barriercost += log(Y.row(dim_hs_top[i]).array().pow(2.0).sum() - Y.middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).array().pow(2.0).sum())/2;}
    if (model->dim_gT) {barriercostTerminal += YT.topRows(model->dim_gT).array().log().sum();}
    for (int i = 0; i < model->dim_hTs.size(); ++i) {barriercostTerminal += log(YT.row(dim_hTs_top[i]).array().pow(2.0).sum() - YT.middleRows(dim_hTs_top[i]+1, model->dim_hTs[i]-1).array().pow(2.0).sum())/2;}
    logcost = cost - param.mu * (barriercost + barriercostTerminal);

    error = 0.0;
    if (model->dim_ec) {error += EC.colwise().lpNorm<1>().sum();}
    if (model->dim_c) {error += (C + Y).colwise().lpNorm<1>().sum();}
    if (model->dim_ecT) {error += ECT.lpNorm<1>();}
    if (model->dim_cT) {error += (CT + YT).lpNorm<1>();}
    error = std::max(param.tolerance, error);
    
    if (error < param.tolerance) {error = 0;}

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

    clock_t start;
    clock_t finish;
    double duration;

    std::cout << std::setw(4) << "iter" 
    << std::setw(3) << "bp" 
    << std::setw(3) << "fp" 
    << std::setw(7) << "mu" 
    << std::setw(12) << "Cost" 
    << std::setw(12) << "OptErr" 
    << std::setw(4) << "Reg" 
    << std::setw(7) << "Step" << std::endl;

    while (iter++ < this->param.max_iter) {
        // std::cout<< "\niter : " << iter << std::endl;

        // std::cout<< "Backward Pass" << std::endl;
        // start = clock();
        this->backwardPass();
        if (backward_failed && regulate==param.max_regularization){
            std::cout << "Max regulation (backward_failed)" << std::endl;
            break;
        }
        if (backward_failed) {
            // std::cout<< "Backward Failed" << std::endl;
            this->logPrint();
            continue;
        }
        // finish = clock();
        // duration = (double)(finish - start) / CLOCKS_PER_SEC;
        // std::cout << duration << "seconds" << std::endl;
        
        // std::cout<< "Forward Pass" << std::endl;
        // start = clock();
        this->forwardPass();
        // finish = clock();
        // duration = (double)(finish - start) / CLOCKS_PER_SEC;
        // std::cout << duration << "seconds" << std::endl;
        
        this->logPrint();
        
        all_cost.push_back(cost);

        // CHECK
        if (opterror <= param.tolerance) {
        // if (std::max(opterror, param.mu) <= param.tolerance) {
            std::cout << "Optimal Solution" << std::endl;
            break;
        }

        if (forward_failed && regulate==param.max_regularization){
            std::cout << "Max regulation (forward_failed)" << std::endl;
            break;
        }

        if (opterror <= 0.2 * param.mu) {
            param.mu = std::max((param.tolerance / 10), std::min(0.2 * param.mu, std::pow(param.mu, 1.2)));
            resetFilter();
            resetRegulation();
        }
    }
}

Eigen::MatrixXd IPDDP::L(const Eigen::VectorXd& x) {
    Eigen::MatrixXd Lx = (x(0) * Eigen::VectorXd::Ones(x.rows())).asDiagonal();
    Lx.col(0) = x;
    Lx.row(0) = x.transpose();
    return Lx;
}

void IPDDP::backwardPass() {
    VectorXdual2nd x(model->dim_x);
    VectorXdual2nd u(model->dim_u);
    Eigen::VectorXd y(model->dim_c);
    Eigen::VectorXd s(model->dim_c);
    Eigen::VectorXd c_v(model->dim_c);

    Eigen::MatrixXd Y_;
    Eigen::MatrixXd S_;

    Eigen::VectorXd Vx(model->dim_rn);
    Eigen::MatrixXd Vxx(model->dim_rn,model->dim_rn);

    Eigen::MatrixXd fx(model->dim_rn,model->dim_rn), fu(model->dim_rn,model->dim_u);
    Eigen::MatrixXd Qsx(model->dim_c,model->dim_rn), Qsu(model->dim_c,model->dim_u);
    // Eigen::Tensor<double, 3> fxx(model->dim_rn,model->dim_rn,model->dim_rn);
    // Eigen::Tensor<double, 3> fxu(model->dim_rn,model->dim_rn,model->dim_u);
    // Eigen::Tensor<double, 3> fuu(model->dim_rn,model->dim_u,model->dim_u);

    Eigen::VectorXd qx(model->dim_rn), qu(model->dim_u);
    Eigen::MatrixXd qdd(model->dim_rn+model->dim_u, model->dim_rn+model->dim_u);
    Eigen::MatrixXd qxx(model->dim_rn,model->dim_rn), qxu(model->dim_rn,model->dim_u), quu(model->dim_u,model->dim_u);

    Eigen::VectorXd Qx(model->dim_rn), Qu(model->dim_u);
    Eigen::MatrixXd Qxx(model->dim_rn,model->dim_rn), Qxu(model->dim_rn,model->dim_u), Quu(model->dim_u,model->dim_u);
    Eigen::MatrixXd Quu_sim(model->dim_u,model->dim_u);

    Eigen::MatrixXd Yinv;
    Eigen::MatrixXd SYinv;

    Eigen::VectorXd rp;
    Eigen::VectorXd rd;
    Eigen::VectorXd r;

    Eigen::LLT<Eigen::MatrixXd> Quu_llt;
    Eigen::MatrixXd R;

    Eigen::VectorXd ku_(model->dim_u);
    Eigen::VectorXd ky_(model->dim_c);
    Eigen::VectorXd ks_(model->dim_c);
    Eigen::MatrixXd Ku_(model->dim_u, model->dim_rn);
    Eigen::MatrixXd Ky_(model->dim_c, model->dim_rn);
    Eigen::MatrixXd Ks_(model->dim_c, model->dim_rn);

    opterror = 0.0;

    dV = Eigen::VectorXd::Zero(2);

    checkRegulate();

    x = X.col(model->N).cast<dual2nd>();
    Vx = model->px(x);
    Vxx = model->pxx(x);

    // CHECK: Inequality Terminal Constraint
    if (model->dim_cT) {
        Eigen::MatrixXd YT_ = Eigen::MatrixXd::Zero(model->dim_cT, model->dim_cT);
        Eigen::MatrixXd ST_ = Eigen::MatrixXd::Zero(model->dim_cT, model->dim_cT);
        if (model->dim_gT) {
            YT_.topLeftCorner(model->dim_gT, model->dim_gT) = y.topRows(model->dim_gT).asDiagonal();
            ST_.topLeftCorner(model->dim_gT, model->dim_gT) = s.topRows(model->dim_gT).asDiagonal();
        }
        for (int i = 0; i < model->dim_hTs.size(); ++i) {
            YT_.block(dim_hTs_top[i], dim_hTs_top[i], model->dim_hTs[i], model->dim_hTs[i]) = L(YT.middleRows(dim_hTs_top[i], model->dim_hTs[i]));
            ST_.block(dim_hTs_top[i], dim_hTs_top[i], model->dim_hTs[i], model->dim_hTs[i]) = L(ST.middleRows(dim_hTs_top[i], model->dim_hTs[i]));
        }
        Eigen::MatrixXd YTinv = YT_.inverse();
        Eigen::MatrixXd STYTinv = YTinv * ST_;

        Eigen::MatrixXd QsxT = model->cTx(x);
        
        Eigen::VectorXd rpT = CT + YT;
        Eigen::VectorXd rdT = YT_*ST - param.mu*eT;
        Eigen::VectorXd rT = ST_*rpT - rdT;
        
        kyT = - rpT;
        KyT = - QsxT;
        ksT = YTinv * rT;
        KsT = STYTinv * QsxT;

        Vx += KsT.transpose() * CT + QsxT.transpose() * ksT;
        Vxx += QsxT.transpose() * KsT + KsT.transpose() * QsxT;
    }

    // TODO: Equality Terminal Constraint
    // if (model->dim_ecT) {

    // }


    backward_failed = false;

    for (int t = model->N - 1; t >= 0; --t) {
        int t_dim_x = t * model->dim_rn;

        x = X.col(t).cast<dual2nd>();
        u = U.col(t).cast<dual2nd>();

        fx = model->fx(x,u);
        fu = model->fu(x,u);

        Qsx = model->cx(x,u);
        Qsu = model->cu(x,u);

        // vectorHessian(fxx, model->f, fs, x, u, "xx");
        // vectorHessian(fxu, model->f, fs, x, u, "xu");
        // vectorHessian(fuu, model->f, fs, x, u, "uu");

        qx = model->qx(x,u);
        qu = model->qu(x,u);

        qdd = model->qdd(x,u);
        qxx = qdd.topLeftCorner(model->dim_rn, model->dim_rn);
        qxu = qdd.block(0, model->dim_rn, model->dim_rn, model->dim_u);
        quu = qdd.bottomRightCorner(model->dim_u, model->dim_u);

        Qx = qx + (Qsx.transpose() * s) + (fx.transpose() * Vx);
        Qu = qu + (Qsu.transpose() * s) + (fu.transpose() * Vx);
        
        // Regularization
        // Qxx = qxx + (fx.transpose() * Vxx * fx);
        // Qxu = qxu + (fx.transpose() * Vxx * fu);
        // Quu = quu + (fu.transpose() * Vxx * fu);
        Qxx = qxx + (fx.transpose() * Vxx * fx);
        // Step 1
        Qxu = qxu + (fx.transpose() * (Vxx + (Eigen::MatrixXd::Identity(model->dim_rn, model->dim_rn) * (std::pow(1.6, regulate) - 1))) * fu);
        Quu = quu + (fu.transpose() * (Vxx + (Eigen::MatrixXd::Identity(model->dim_rn, model->dim_rn) * (std::pow(1.6, regulate) - 1))) * fu);
        // Step 2
        Qxx += Eigen::MatrixXd::Identity(model->dim_rn, model->dim_rn) * (std::pow(1.6, regulate) - 1);
        Quu += Eigen::MatrixXd::Identity(model->dim_u, model->dim_u) * (std::pow(1.6, regulate) - 1);
        
        // iLQR to DDP (TODO: Vector-Hessian Product)
        // Qxx = qxx + (fx.transpose() * Vxx * fx) + tensdot(Vx, fxx);
        // Qxu = qxu + (fx.transpose() * Vxx * fu) + tensdot(Vx, fxu);
        // Quu = quu + (fu.transpose() * Vxx * fu) + tensdot(Vx, fuu);

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
            
            rp = c_v + y;
            rd = Y_*s - param.mu*e;
            r = S_*rp - rd;

            Qx += Qsx.transpose() * (Yinv * r);
            Qu += Qsu.transpose() * (Yinv * r);
    
            Quu += Qsu.transpose() * SYinv * Qsu;
            Qxu += Qsx.transpose() * SYinv * Qsu;
            Qxx += Qsx.transpose() * SYinv * Qsx;
        }

        // TODO
        // if (model->dim_ec) {

        // }

        Quu_sim = 0.5*(Quu + Quu.transpose());
        Quu = Quu_sim;
        Quu_llt = Eigen::LLT<Eigen::MatrixXd>(Quu);
        if (!Quu.isApprox(Quu.transpose()) || Quu_llt.info() == Eigen::NumericalIssue) {
            backward_failed = true;
            break;
        }
        R = Quu_llt.matrixU();

        ku_ = -R.inverse() * (R.transpose().inverse() * Qu);
        Ku_ = -R.inverse() * (R.transpose().inverse() * Qxu.transpose());
        
        dV(0) += ku_.transpose() * Qu;
        dV(1) += 0.5 * ku_.transpose() * Quu * ku_;
        
        Vx = Qx + (Ku_.transpose() * Qu) + (Ku_.transpose() * Quu * ku_) + (Qxu * ku_);
        Vxx = Qxx + (Ku_.transpose() * Qxu.transpose()) + (Qxu * Ku_) + (Ku_.transpose() * Quu * Ku_);
        
        ku.col(t) = ku_;
        Ku.middleCols(t_dim_x, model->dim_rn) = Ku_;

        opterror = std::max({Qu.lpNorm<Eigen::Infinity>(), opterror});

        // CHECK: Value Update with Lagrangian (s / Typo in Paper (transpose K and R))
        if (model->dim_c) {
            ks_ = (Yinv * r) + (SYinv * Qsu * ku_);
            Ks_ = SYinv * (Qsx + Qsu * Ku_);
            ky_ = -rp - Qsu * ku_;
            Ky_ = -Qsx - Qsu * Ku_;

            Vx += (Ks_.transpose() * c_v) + (Qsx.transpose() * ks_) + (Ku_.transpose() * Qsu.transpose() * ks_) + (Ks_.transpose() * Qsu * ku_);
            Vxx += (Qsx.transpose() * Ks_) + (Ks_.transpose() * Qsx) + (Ku_.transpose() * Qsu.transpose() * Ks_) + (Ks_.transpose() * Qsu * Ku_);

            ks.col(t) = ks_;
            Ks.middleCols(t_dim_x, model->dim_rn) = Ks_;
            ky.col(t) = ky_;
            Ky.middleCols(t_dim_x, model->dim_rn) = Ky_;

            opterror = std::max({rp.lpNorm<Eigen::Infinity>(), rd.lpNorm<Eigen::Infinity>(), opterror});
        }

        // TODO
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
    Eigen::MatrixXd EC_new(model->dim_ec, model->N);

    Eigen::VectorXd YT_new(model->dim_cT);
    Eigen::VectorXd ST_new(model->dim_cT);
    Eigen::VectorXd CT_new(model->dim_cT);
    Eigen::VectorXd ECT_new(model->dim_ecT);

    double tau = std::max(0.99, 1.0 - param.mu);
    
    double cost_new = 0.0;
    double logcost_new = 0.0;
    double barriercost_new = 0.0;
    double barriercostTerminal_new = 0.0;
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
        // TODO
        // if (model->dim_ec) {

        // }
        if (model->dim_cT) {
            Eigen::VectorXd dxT = model->perturb(X_new.col(model->N), X.col(model->N));
            YT_new = YT + (step_size * kyT) + KyT * dx;
            ST_new = ST + (step_size * ksT) + KsT * dx;
        }
        // TODO
        // if (model->dim_ecT) {

        // }

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
        // TODO
        // if (model->dim_ec) {

        // }
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
        // TODO
        // if (model->dim_ecT) {

        // }
        
        if (forward_failed) {continue;}

        // Cost
        cost_new = calculateTotalCost(X_new, U_new);
        if (model->dim_g) {barriercost_new += Y_new.topRows(model->dim_g).array().log().sum();}
        for (int i = 0; i < model->dim_hs.size(); ++i) {barriercost_new += log(Y_new.row(dim_hs_top[i]).array().pow(2.0).sum() - Y_new.middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).array().pow(2.0).sum())/2;}
        if (model->dim_gT) {barriercostTerminal_new += YT_new.topRows(model->dim_gT).array().log().sum();}
        for (int i = 0; i < model->dim_hTs.size(); ++i) {barriercostTerminal_new += log(YT_new.row(dim_hTs_top[i]).array().pow(2.0).sum() - YT_new.middleRows(dim_hTs_top[i]+1, model->dim_hTs[i]-1).array().pow(2.0).sum())/2;}
        logcost_new = cost_new - param.mu * (barriercost_new + barriercostTerminal_new);
        
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
        if (model->dim_cT) {CT_new = model->cT(X.col(model->N)).cast<double>();}
        if (model->dim_ecT) {ECT_new = model->ecT(X.col(model->N)).cast<double>();}
        
        error_new = 0.0;
        if (model->dim_ec) {error_new += EC_new.colwise().lpNorm<1>().sum();}
        if (model->dim_c) {error_new += (C_new + Y_new).colwise().lpNorm<1>().sum();}
        if (model->dim_ecT) {error_new += ECT_new.lpNorm<1>();}
        if (model->dim_cT) {error_new += (CT_new + YT_new).lpNorm<1>();}
        error_new = std::max(param.tolerance, error_new);

        // With Expected Value Decrement
        // dV_act = logcost - logcost_new;
        // dV_exp = step * dV(0) + step * step * dV(1);
        // if ((1e-4 * dV_exp < dV_act && dV_act < 10 * dV_exp) && error >= error_new) {break;}
        // Original
        if (logcost >= logcost_new && error >= error_new) {break;}
        
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
        if (model->dim_ec) {EC = EC_new;}
        if (model->dim_cT) {
            YT = YT_new;
            ST = ST_new;
            CT = CT_new;
        }
        if (model->dim_ecT) {ECT = ECT_new;}
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
              << std::setw(3) << backward_failed 
              << std::setw(3) << forward_failed 
              << std::setw(7) << param.mu 
              << std::setw(12) << cost 
              << std::setw(12) << opterror 
              << std::setw(4) << regulate 
              << std::setw(7) << step_list[step] << std::endl;
}