#pragma once

#define MAX_STEP 20

#include "param.h"
#include "model_base.h"
#include "helper_function.h"

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
    std::vector<int> dim_hs_top;

    Eigen::MatrixXd X;
    Eigen::MatrixXd U;
    Eigen::MatrixXd Y;
    Eigen::MatrixXd S;
    Eigen::MatrixXd C;

    double cost;
    Param param;
    void initialRoll();
    void resetFilter();
    double logcost;
    double error;
    
    std::vector<double> step_list;
    int step;
    int forward_failed;

    void resetRegulation();
    int regulate;
    bool backward_failed;

    Eigen::MatrixXd ku;
    Eigen::MatrixXd ky;
    Eigen::MatrixXd ks;
    Eigen::MatrixXd Ku;
    Eigen::MatrixXd Ky;
    Eigen::MatrixXd Ks;

    Eigen::VectorXd e;

    double opterror;
    // Eigen::VectorXd dV;

    std::vector<double> all_cost;

    // Algorithm
    Eigen::MatrixXd L(const Eigen::VectorXd& x);
    void backwardPass();
    void checkRegulate();
    void forwardPass();
    double calculateTotalCost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U);
};

template<typename ModelClass>
IPDDP::IPDDP(std::shared_ptr<ModelClass> model_ptr) : model(model_ptr) {
    static_assert(std::is_base_of<ModelBase, ModelClass>::value, "ModelClass must be derived from ModelBase");
    // Stack Constraint (TODO: Move to Model)
    model->dim_c = model->dim_g + accumulate(model->dim_hs.begin(), model->dim_hs.end(), 0);
    if (!model->dim_rn) {model->dim_rn = model->dim_x;}
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

    // Initialization
    if (model->X_init.size()) {X = model->X_init;}
    else {X = Eigen::MatrixXd::Zero(model->dim_x, model->N+1);}
    if (model->U_init.size()) {U = model->U_init;}
    else {U = Eigen::MatrixXd::Zero(model->dim_u, model->N);}

    if (model->Y_init.size()) {Y = model->Y_init;}
    else {
        Y = Eigen::MatrixXd::Zero(model->dim_c, model->N);
        if (model->dim_g) {Y.topRows(model->dim_g) = Eigen::MatrixXd::Ones(model->dim_g,model->N);}
        for (auto dim_h_top : dim_hs_top) {Y.row(dim_h_top) = Eigen::VectorXd::Ones(model->N);}
    }

    if (model->S_init.size()) {S = model->S_init;}
    else {
        S = Eigen::MatrixXd::Zero(model->dim_c, model->N);
        if (model->dim_g) {S.topRows(model->dim_g) = 0.01*Eigen::MatrixXd::Ones(model->dim_g,model->N);}
        for (auto dim_h_top : dim_hs_top) {S.row(dim_h_top) = 0.01*Eigen::VectorXd::Ones(model->N);}
    }
    
    ku.resize(model->dim_u, model->N);
    ky.resize(model->dim_c, model->N);
    ks.resize(model->dim_c, model->N);
    Ku.resize(model->dim_u, model->dim_rn * model->N);
    Ky.resize(model->dim_c, model->dim_rn * model->N);
    Ks.resize(model->dim_c, model->dim_rn * model->N);

    e = Eigen::VectorXd::Ones(model->dim_c);
    for (int i = 0; i < model->dim_hs.size(); ++i) {
        e.middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1) = Eigen::VectorXd::Zero(model->dim_hs[i]-1);
    }
}

IPDDP::~IPDDP() {
}

void IPDDP::init(Param param) {
    this->param = param;

    this->initialRoll();
    if (this->param.mu == 0) {this->param.mu = cost / model->N / model->dim_c;} // Auto Select
    this->resetFilter();
    this->resetRegulation();

    for (int i = 0; i <= MAX_STEP; ++i) {
        step_list.push_back(std::pow(2.0, static_cast<double>(-i)));
    }
}

void IPDDP::initialRoll() {
    this->C.resize(this->model->dim_c, this->model->N);
    for (int t = 0; t < this->model->N; ++t) {
        C.col(t) = model->c(X.col(t), U.col(t)).cast<double>();
        X.col(t+1) = model->f(X.col(t), U.col(t)).cast<double>();
    }
    cost = calculateTotalCost(X, U);
}

void IPDDP::resetFilter() {
    double barriercost = 0.0;
    if (model->dim_g) {barriercost += Y.topRows(model->dim_g).array().log().sum();}
    for (int i = 0; i < model->dim_hs.size(); ++i) {barriercost += log(Y.row(dim_hs_top[i]).array().pow(2.0).sum() - Y.middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).array().pow(2.0).sum())/2;}
    logcost = cost - param.mu * barriercost;
    error = (C + Y).colwise().lpNorm<1>().sum();
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
    int iter = 0;

    clock_t start;
    clock_t finish;
    double duration;

    while (iter++ < this->param.max_iter) {
        std::cout<< "\niter : " << iter << std::endl;

        std::cout<< "Backward Pass" << std::endl;
        // start = clock();
        this->backwardPass();
        if (backward_failed && regulate==24){
            std::cout << "Max regulation (backward_failed)" << std::endl;
            break;
        }
        if (backward_failed) {
            std::cout<< "Backward Failed" << std::endl;
            continue;
        }
        // finish = clock();
        // duration = (double)(finish - start) / CLOCKS_PER_SEC;
        // std::cout << duration << "seconds" << std::endl;
        
        std::cout<< "Forward Pass" << std::endl;
        // start = clock();
        this->forwardPass();
        // finish = clock();
        // duration = (double)(finish - start) / CLOCKS_PER_SEC;
        // std::cout << duration << "seconds" << std::endl;
        
        std::cout<< "mu : " << param.mu << std::endl;
        std::cout<< "Cost : " << cost << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout<< "Opt Error : " << opterror << std::endl;
        std::cout<< "Regulate : " << regulate << std::endl;
        std::cout<< "Step Size : " << step_list[step] << std::endl;
        all_cost.push_back(cost);

        // CHECK
        if (opterror <= param.tolerance) {
        // if (std::max(opterror, param.mu) <= param.tolerance) {
            std::cout << "Optimal Solution" << std::endl;
            break;
        }

        if (forward_failed && regulate==24){
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
    Eigen::MatrixXd Lx = (x(0)*Eigen::VectorXd::Ones(x.rows())).asDiagonal();
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

    // while (true) {
        // dV = Eigen::VectorXd::Zero(2);

        checkRegulate();

        x = X.col(model->N).cast<dual2nd>();
        Vx = model->px(x);
        Vxx = model->pxx(x);

        // CHECK
        backward_failed = false;
        // std::cout<<"Y = "<<Y<<std::endl;
        // std::cout<<"S = "<<S<<std::endl;

        for (int t = model->N - 1; t >= 0; --t) {
            int t_dim_x = t * model->dim_rn;

            x = X.col(t).cast<dual2nd>();
            u = U.col(t).cast<dual2nd>();

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

            // std::cout<<"Y\n"<<Y_<<std::endl;
            // std::cout<<"Y_inv\n"<<Y_.inverse()<<std::endl;

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

            Qxx = qxx + (fx.transpose() * Vxx * fx);
            // CHECK: Regularization
            // Qxu = qxu + (fx.transpose() * Vxx * fu);
            // Quu = quu + (fu.transpose() * Vxx * fu);
            Qxu = qxu + (fx.transpose() * (Vxx + (Eigen::MatrixXd::Identity(model->dim_rn, model->dim_rn) * (std::pow(1.6, regulate) - 1))) * fu);
            Quu = quu + (fu.transpose() * (Vxx + (Eigen::MatrixXd::Identity(model->dim_rn, model->dim_rn) * (std::pow(1.6, regulate) - 1))) * fu);

            // Qxx = qxx + (fx.transpose() * Vxx * fx) + tensdot(Vx, fxx);
            // Qxu = qxu + (fx.transpose() * Vxx * fu) + tensdot(Vx, fxu);
            // Quu = quu + (fu.transpose() * Vxx * fu) + tensdot(Vx, fuu);

            Yinv = Y_.inverse();
            SYinv = Yinv * S_;

            rp = c_v + y;
            rd = Y_*s - param.mu*e;
            r = S_*rp - rd;

            Quu += Qsu.transpose() * SYinv * Qsu;
            Qxu += Qsx.transpose() * SYinv * Qsu;
            Qxx += Qsx.transpose() * SYinv * Qsx;

            Qx += Qsx.transpose() * (Yinv * r);
            Qu += Qsu.transpose() * (Yinv * r);

            Quu += Eigen::MatrixXd::Identity(model->dim_u, model->dim_u) * (std::pow(1.6, regulate) - 1);
            // Quu += Eigen::MatrixXd::Identity(model->dim_u, model->dim_u) * regulate;
            // Quu += quu * (std::pow(1.6, regulate) - 1);

            Quu_sim = 0.5*(Quu + Quu.transpose());
            // std::cout<<"Quu_sim: "<<Quu_sim<<std::endl;
            Quu = Quu_sim;
            Quu_llt = Eigen::LLT<Eigen::MatrixXd>(Quu);
            if (!Quu.isApprox(Quu.transpose()) || Quu_llt.info() == Eigen::NumericalIssue) {
                backward_failed = true;
                // std::cout<<"regulate: "<<regulate<<std::endl;
                // std::cout<<"Quu: "<<Quu<<std::endl;
                break;
            }
            R = Quu_llt.matrixU();

            ku_ = -R.inverse() * (R.transpose().inverse() * Qu);
            Ku_ = -R.inverse() * (R.transpose().inverse() * Qxu.transpose());
            ks_ = (Yinv * r) + (SYinv * Qsu * ku_);
            Ks_ = SYinv * (Qsx + Qsu * Ku_);
            ky_ = -rp - Qsu * ku_;
            Ky_ = -Qsx - Qsu * Ku_;
            
            // dV(0) = dV(0) + (ku_.transpose() * Qu)(0);
            // dV(1) = dV(1) + (0.5 * ku_.transpose() * Quu * ku_)(0);

            Vx = Qx + (Ku_.transpose() * Qu) + (Ku_.transpose() * Quu * ku_) + (Qxu * ku_);
            Vxx = Qxx + (Ku_.transpose() * Qxu.transpose()) + (Qxu * Ku_) + (Ku_.transpose() * Quu * Ku_);

            // CHECK: Value Update with Lagrangian (s / Typo in Paper (transpose K and R))
            Vx += (Ks_.transpose() * c_v) + (Qsx.transpose() * ks_) + (Ku_.transpose() * Qsu.transpose() * ks_) + (Ks_.transpose() * Qsu * ku_);
            Vxx += (Qsx.transpose() * Ks_) + (Ks_.transpose() * Qsx) + (Ku_.transpose() * Qsu.transpose() * Ks_) + (Ks_.transpose() * Qsu * Ku_);
            
            ku.col(t) = ku_;
            Ku.middleCols(t_dim_x, model->dim_rn) = Ku_;
            ks.col(t) = ks_;
            Ks.middleCols(t_dim_x, model->dim_rn) = Ks_;
            ky.col(t) = ky_;
            Ky.middleCols(t_dim_x, model->dim_rn) = Ky_;

            // std::cout<<"ku_: "<<ku_<<std::endl;
            // std::cout<<"Ku_: "<<Ku_<<std::endl;

            // std::cout<<"Qu.lpNorm<Eigen::Infinity>(): "<<Qu.lpNorm<Eigen::Infinity>()<<std::endl;
            // std::cout<<"rp.lpNorm<Eigen::Infinity>(): "<<rp.lpNorm<Eigen::Infinity>()<<std::endl;
            // std::cout<<"rd.lpNorm<Eigen::Infinity>(): "<<rd.lpNorm<Eigen::Infinity>()<<std::endl;

            opterror = std::max({Qu.lpNorm<Eigen::Infinity>(), rp.lpNorm<Eigen::Infinity>(), rd.lpNorm<Eigen::Infinity>(), opterror});
        }
}

void IPDDP::checkRegulate() {
    if (forward_failed || backward_failed) {++regulate;}
    // else if (step == 0) {--regulate;}
    // else if (step <= 3) {regulate = regulate;}
    else {--regulate;}

    if (regulate < 0) {regulate = 0;}
    else if (24 < regulate) {regulate = 24;}
}

void IPDDP::forwardPass() {
    Eigen::VectorXd dx;
    Eigen::MatrixXd X_new(model->dim_x, model->N+1);
    Eigen::MatrixXd U_new(model->dim_u, model->N);
    Eigen::MatrixXd Y_new(model->dim_c, model->N);
    Eigen::MatrixXd S_new(model->dim_c, model->N);
    Eigen::MatrixXd C_new(model->dim_c, model->N);

    double tau = std::max(0.99, 1.0 - param.mu);
    
    double cost_new = 0.0;
    double logcost_new = 0.0;
    double barrier_g_new = 0.0;
    double barrier_h_new = 0.0;
    double error_new = 0.0;

    for (step = 0; step < MAX_STEP; ++step) {

        forward_failed = false;
        double step_size = step_list[step];

        X_new.col(0) = X.col(0);
        for (int t = 0; t < model->N; ++t) {
            int t_dim_x = t * model->dim_rn;
            dx = model->perturb(X_new.col(t), X.col(t));
            Y_new.col(t) = Y.col(t) + (step_size * ky.col(t)) + Ky.middleCols(t_dim_x, model->dim_rn) * dx;
            S_new.col(t) = S.col(t) + (step_size * ks.col(t)) + Ks.middleCols(t_dim_x, model->dim_rn) * dx;
            U_new.col(t) = U.col(t) + (step_size * ku.col(t)) + Ku.middleCols(t_dim_x, model->dim_rn) * dx;
            X_new.col(t+1) = model->f(X_new.col(t), U_new.col(t)).cast<double>();
        }

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

        if (forward_failed) {continue;}

        cost_new = calculateTotalCost(X_new, U_new);

        barrier_g_new = 0.0;
        barrier_h_new = 0.0;

        if (model->dim_g) {barrier_g_new = Y_new.topRows(model->dim_g).array().log().sum();}
        for (int i = 0; i < model->dim_hs.size(); ++i) {barrier_h_new += log(Y_new.row(dim_hs_top[i]).array().pow(2.0).sum() - Y_new.middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).array().pow(2.0).sum())/2;}
        logcost_new = cost_new - param.mu * (barrier_g_new + barrier_h_new);
        for (int t = 0; t < model->N; ++t) {
            C_new.col(t) = model->c(X_new.col(t), U_new.col(t)).cast<double>();
        }
        // std::cout<<"barriercost G = "<<barrier_g_new<<std::endl;
        // std::cout<<"barriercost H = "<<barrier_h_new<<std::endl;
        // error_new = (C + Y).colwise().lpNorm<1>().sum();
        error_new = std::max(param.tolerance, (C_new + Y_new).lpNorm<1>());
        if (logcost >= logcost_new && error >= error_new) {break;}
        // if (logcost >= logcost_new && error >= error_new) {std::cout<<"10"<<std::endl;break;}
        // std::cout<<"error = "<<error<<std::endl;
        forward_failed = true;
    }

    if (!forward_failed) {
        cost = cost_new;
        logcost = logcost_new;
        error = error_new;
        X = X_new;
        U = U_new;
        Y = Y_new;
        S = S_new;
        C = C_new;
        std::cout<<"Y = "<<Y.transpose()<<std::endl;
    }
    else {std::cout<<"Forward Failed"<<std::endl;}
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

