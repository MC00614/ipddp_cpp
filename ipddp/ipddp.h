#pragma once

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

class IPDDP {
public:
    template<typename ModelClass>
    IPDDP(ModelClass model);
    ~IPDDP();

    void init(Param param);
    void solve();

    Eigen::MatrixXd getInitX();
    Eigen::MatrixXd getInitU();
    Eigen::MatrixXd getResX();
    Eigen::MatrixXd getResU();
    std::vector<double> getAllCost();

private:
    double duration_;

    Eigen::MatrixXd X_init;
    Eigen::MatrixXd U_init;

    int N;
    int dim_x;
    int dim_u;
    int dim_g;
    int dim_h;
    int dim_c;
    Eigen::MatrixXd X;
    Eigen::MatrixXd U;
    Eigen::MatrixXd Y;
    Eigen::MatrixXd S;
    Eigen::MatrixXd C;
    // Discrete Time System
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> f;
    // Stage Cost Function
    std::function<dual2nd(VectorXdual2nd, VectorXdual2nd)> q;
    // Terminal Cost Function
    std::function<dual2nd(VectorXdual2nd)> p;
    // Constraint
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> c;

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
IPDDP::IPDDP(ModelClass model) {
    // Check Model
    if (!model.N || !model.dim_x || !model.dim_u) {throw std::invalid_argument("Model Parameter is null.");}
    this->N = model.N;
    this->dim_x = model.dim_x;
    this->dim_u = model.dim_u;
    this->dim_g = model.dim_g;
    this->dim_h = model.dim_h;
    this->dim_c = model.dim_c;

    if (!model.X.size() || !model.U.size()) {throw std::invalid_argument("Model State is null.");}
    this->X = model.X;
    this->U = model.U;
    this->Y = model.Y;
    this->S = model.S;
    
    if (!model.f || !model.q || !model.p) {throw std::invalid_argument("Model Function is null.");}
    this->f = model.f;
    this->q = model.q;
    this->p = model.p;
    this->c = model.c;

    this->ku.resize(this->dim_u, this->N);
    this->ky.resize(this->dim_c, this->N);
    this->ks.resize(this->dim_c, this->N);
    this->Ku.resize(this->dim_u, this->dim_x * this->N);
    this->Ky.resize(this->dim_c, this->dim_x * this->N);
    this->Ks.resize(this->dim_c, this->dim_x * this->N);
}

IPDDP::~IPDDP() {
}

void IPDDP::init(Param param) {
    this->param = param;

    this->initialRoll();
    if (this->param.mu == 0) {this->param.mu = cost / N / dim_c;} // Auto Select
    this->resetFilter();
    this->resetRegulation();

    for (double i = 1; i < 11; ++i) {
        step_list.push_back(std::pow(2.0, -i));
    }
}

void IPDDP::initialRoll() {
    this->C.resize(this->dim_c, this->N);
    for (int t = 0; t < this->N; ++t) {
        C.col(t) = c(X.col(t), U.col(t)).cast<double>();
        X.col(t+1) = f(X.col(t), U.col(t)).cast<double>();
    }
    X_init = X;
    U_init = U;
    cost = calculateTotalCost(X, U);
}

void IPDDP::resetFilter() {
    double barriercost = 0.0;
    if (dim_g) {barriercost += Y.topRows(dim_g).array().log().sum();}
    if (dim_h) {barriercost += log(Y.row(dim_c-dim_h).array().pow(2.0).sum() - Y.bottomRows(dim_h-1).array().pow(2.0).sum())/2;}
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
    for (int t = 0; t < N; ++t) {
        cost += q(X.col(t), U.col(t));
    }
    cost += p(X.col(N));
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
        if (backward_failed) {continue;}
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

        // CHECK
        if (opterror < 10000000*param.mu) {
        // if (opterror <= (0.2 * param.mu)) {
            param.mu = std::max((param.tolerance / 10), std::min(0.2 * param.mu, std::pow(param.mu, 1.2)));
            resetFilter();
            resetRegulation();
        }
    }
}

Eigen::MatrixXd IPDDP::L(const Eigen::VectorXd& x) {
    Eigen::MatrixXd Lx = (x(0)*Eigen::VectorXd::Ones(x.rows())).asDiagonal();
    Lx.leftCols(1) = x;
    Lx.topRows(1) = x.transpose();
    return Lx;
}

void IPDDP::backwardPass() {
    VectorXdual2nd x(dim_x);
    VectorXdual2nd u(dim_u);
    Eigen::VectorXd y(dim_c);
    Eigen::VectorXd s(dim_c);
    Eigen::VectorXd c_v(dim_c);

    Eigen::VectorXd Vx(dim_x);
    Eigen::MatrixXd Vxx(dim_x,dim_x);

    Eigen::MatrixXd fx(dim_x,dim_x), fu(dim_x,dim_u);
    Eigen::MatrixXd Qsx(dim_c,dim_x), Qsu(dim_c,dim_u);
    // Eigen::Tensor<double, 3> fxx(dim_x,dim_x,dim_x);
    // Eigen::Tensor<double, 3> fxu(dim_x,dim_x,dim_u);
    // Eigen::Tensor<double, 3> fuu(dim_x,dim_u,dim_u);

    Eigen::VectorXd qx(dim_x), qu(dim_u);
    Eigen::MatrixXd qxx(dim_x,dim_x), qxu(dim_x,dim_u), quu(dim_u,dim_u);

    Eigen::VectorXd Qx(dim_x), Qu(dim_u);
    Eigen::MatrixXd Qxx(dim_x,dim_x), Qxu(dim_x,dim_u), Quu(dim_u,dim_u);
    Eigen::MatrixXd Quu_sim(dim_u,dim_u);


    // residual loc.

    
    Eigen::LLT<Eigen::MatrixXd> Quu_llt;
    Eigen::MatrixXd R;
    Eigen::MatrixXd row1;
    Eigen::MatrixXd row2;

    Eigen::MatrixXd kK(dim_u, 1 + dim_x);

    Eigen::VectorXd ku_(dim_u);
    Eigen::VectorXd ky_(dim_c);
    Eigen::VectorXd ks_(dim_c);
    Eigen::MatrixXd Ku_(dim_u, dim_x);
    Eigen::MatrixXd Ky_(dim_c, dim_x);
    Eigen::MatrixXd Ks_(dim_c, dim_x);

    opterror = 0.0;

    // while (true) {
        // dV = Eigen::VectorXd::Zero(2);

        checkRegulate();

        x = X.col(N).cast<dual2nd>();
        Vx = gradient(p, wrt(x), at(x));
        Vxx = hessian(p, wrt(x), at(x));

        // CHECK
        backward_failed = 0;

        for (int t = N - 1; t >= 0; --t) {
            int t_dim_x = t * dim_x;

            x = X.col(t).cast<dual2nd>();
            u = U.col(t).cast<dual2nd>();

            y = Y.col(t);
            s = S.col(t);
            c_v = C.col(t);

            Eigen::MatrixXd Y_ = Eigen::MatrixXd::Zero(dim_c, dim_c);
            Eigen::MatrixXd S_ = Eigen::MatrixXd::Zero(dim_c, dim_c);
            if (dim_g) {
                Y_.topLeftCorner(dim_g, dim_g) = y.topRows(dim_g).asDiagonal();
                S_.topLeftCorner(dim_g, dim_g) = s.topRows(dim_g).asDiagonal();
            }
            if (dim_h) {
                Y_.bottomRightCorner(dim_h, dim_h) = L(y.bottomRows(dim_h));
                S_.bottomRightCorner(dim_h, dim_h) = L(s.bottomRows(dim_h));
            }
            Eigen::VectorXd e = Eigen::VectorXd::Ones(dim_c);
            if (dim_h) {
                e.bottomRows(dim_h-1) = Eigen::VectorXd::Zero(dim_h-1);
            }

            fx = jacobian(f, wrt(x), at(x,u));
            fu = jacobian(f, wrt(u), at(x,u));

            Qsx = jacobian(c, wrt(x), at(x,u));
            Qsu = jacobian(c, wrt(u), at(x,u));

            // vectorHessian(fxx, f, fs, x, u, "xx");
            // vectorHessian(fxu, f, fs, x, u, "xu");
            // vectorHessian(fuu, f, fs, x, u, "uu");

            qx = gradient(q, wrt(x), at(x,u));
            qu = gradient(q, wrt(u), at(x,u));

            Qx = qx + (Qsx.transpose() * s) + (fx.transpose() * Vx);
            Qu = qu + (Qsu.transpose() * s) + (fu.transpose() * Vx);

            qxx = hessian(q, wrt(x), at(x,u));
            scalarHessian(qxu, q, x, u, "xu");
            quu = hessian(q, wrt(u), at(x,u));

            Qxx = qxx + (fx.transpose() * Vxx * fx);
            Qxu = qxu + (fx.transpose() * Vxx * fu);
            Quu = quu + (fu.transpose() * Vxx * fu);
            // std::cout<<"Quu1 = "<<Quu<<std::endl;

            // Qxx = qxx + (fx.transpose() * Vxx * fx) + tensdot(Vx, fxx);
            // Qxu = qxu + (fx.transpose() * Vxx * fu) + tensdot(Vx, fxu);
            // Quu = quu + (fu.transpose() * Vxx * fu) + tensdot(Vx, fuu);

            Eigen::MatrixXd Yinv = Y_.inverse();
            Eigen::MatrixXd SYinv = Yinv * S_;

            Eigen::VectorXd rp = c_v + y;
            Eigen::VectorXd rd = Y_*s - param.mu*e;
            Eigen::VectorXd r = S_*rp - rd;

            Quu += Qsu.transpose() * SYinv * Qsu;
            Qxu += Qsx.transpose() * SYinv * Qsu;
            Qxx += Qsx.transpose() * SYinv * Qsx;

            Qx += Qsx.transpose() * (Yinv * r);
            Qu += Qsu.transpose() * (Yinv * r);

            Quu += Eigen::MatrixXd::Identity(dim_u, dim_u) * (std::pow(1.6, regulate) - 1);
            // Quu += Eigen::MatrixXd::Identity(dim_u, dim_u) * regulate;
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
            
            ku.col(t) = ku_;
            Ku.middleCols(t_dim_x, dim_x) = Ku_;
            ks.col(t) = ks_;
            Ks.middleCols(t_dim_x, dim_x) = Ks_;
            ky.col(t) = ky_;
            Ky.middleCols(t_dim_x, dim_x) = Ky_;

            // std::cout<<"ku_: "<<ku_<<std::endl;
            // std::cout<<"Ku_: "<<Ku_<<std::endl;

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
    Eigen::MatrixXd X_new(dim_x, N+1);
    Eigen::MatrixXd U_new(dim_u, N);
    Eigen::MatrixXd Y_new(dim_c, N);
    Eigen::MatrixXd S_new(dim_c, N);
    Eigen::MatrixXd C_new(dim_c, N);

    double tau = std::max(0.99, 1.0 - param.mu);
    
    double cost_new = 0.0;
    double logcost_new = 0.0;
    double barrier_g_new = 0.0;
    double barrier_h_new = 0.0;
    double error_new = 0.0;

    for (step = 0; step < step_list.size(); ++step) {

        forward_failed = false;
        double step_size = step_list[step];

        X_new.col(0) = X.col(0);
        for (int t = 0; t < N; ++t) {
            int t_dim_x = t * dim_x;
            Y_new.col(t) = Y.col(t) + (step_size * ky.col(t)) + (Ky.middleCols(t_dim_x, dim_x) * (X_new.col(t) - X.col(t)));
            S_new.col(t) = S.col(t) + (step_size * ks.col(t)) + (Ks.middleCols(t_dim_x, dim_x) * (X_new.col(t) - X.col(t)));
            if (dim_g) {
                if ((Y_new.col(t).topRows(dim_g).array() < (1 - tau) * Y.col(t).topRows(dim_g).array()).any()) {forward_failed = true; break;}
                if ((S_new.col(t).topRows(dim_g).array() < (1 - tau) * S.col(t).topRows(dim_g).array()).any()) {forward_failed = true; break;}
            }
            if (dim_h) {
                if ((Y_new.col(t).row(dim_c-dim_h).array().pow(2.0) - Y_new.col(t).bottomRows(dim_h-1).array().pow(2.0).sum()
                < (1 - tau) * (Y.col(t).row(dim_c-dim_h).array().pow(2.0) - Y.col(t).bottomRows(dim_h-1).array().pow(2.0).sum())).any()) {forward_failed = true; break;}
                if ((S_new.col(t).row(dim_c-dim_h).array().pow(2.0) - S_new.col(t).bottomRows(dim_h-1).array().pow(2.0).sum()
                < (1 - tau) * (S.col(t).row(dim_c-dim_h).array().pow(2.0) - S.col(t).bottomRows(dim_h-1).array().pow(2.0).sum())).any()) {forward_failed = true; break;}
            }
            U_new.col(t) = U.col(t) + (step_size * ku.col(t)) + (Ku.middleCols(t_dim_x, dim_x) * (X_new.col(t) - X.col(t)));
            X_new.col(t+1) = f(X_new.col(t), U_new.col(t)).cast<double>();
        }

        if (forward_failed) {continue;}

        cost_new = calculateTotalCost(X_new, U_new);

        if (dim_g) {barrier_g_new = Y_new.topRows(dim_g).array().log().sum();}
        if (dim_h) {barrier_h_new += log(Y_new.row(dim_c-dim_h).array().pow(2.0).sum() - Y_new.bottomRows(dim_h-1).array().pow(2.0).sum())/2;}
        // std::cout<<"barriercost G = "<<barrier_g_new<<std::endl;
        // std::cout<<"barriercost H = "<<barrier_h_new<<std::endl;
        logcost_new = cost_new - param.mu * (barrier_g_new + barrier_h_new);
        for (int t = 0; t < N; ++t) {
            C_new.col(t) = c(X_new.col(t), U_new.col(t)).cast<double>();
        }
        error = (C + Y).colwise().lpNorm<1>().sum();
        // error_new = std::max(param.tolerance, (C_new + Y_new).lpNorm<1>());
        // if (logcost >= logcost_new || error >= error_new) {break;}
        if (logcost >= logcost_new && error >= error_new) {break;}
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
        // std::cout<<"Y = "<<Y.transpose()<<std::endl;
    }
    else {std::cout<<"Forward Failed"<<std::endl;}
}

Eigen::MatrixXd IPDDP::getInitX() {
    return X_init;
}

Eigen::MatrixXd IPDDP::getInitU() {
    return U_init;
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