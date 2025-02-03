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
    ModelBase model;

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
IPDDP::IPDDP(ModelClass model_) : model(model_) {
    // Stack Constraint
    model.dim_c = model.dim_g + model.dim_h + model.dim_h2;
    model.c = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd c_n(model.dim_c);
        if (model.dim_g) {c_n.topRows(model.dim_g) = model.g(x, u);}
        if (model.dim_h) {c_n.middleRows(model.dim_g, model.dim_h) = model.h(x, u);}
        if (model.dim_h2) {c_n.bottomRows(model.dim_h2) = model.h2(x, u);}
        return c_n;
    };

    // Initialization
    if (model.X_init.size()) {X = model.X_init;}
    else if (model.x0.size()) {
        X = Eigen::MatrixXd::Zero(model.dim_x, model.N+1);
        X.col(0) = model.x0;
    }
    else {
        std::cout<<"Warning: Initial State Not Found"<<std::endl;
        X = Eigen::MatrixXd::Zero(model.dim_x, model.N+1);
    }
    if (model.U_init.size()) {U = model.U_init;}
    else {U = Eigen::MatrixXd::Zero(model.dim_u, model.N);}

    if (model.Y_init.size()) {Y = model.Y_init;}
    else {
        Y = Eigen::MatrixXd::Zero(model.dim_c, model.N);
        if (model.dim_g) {Y.topRows(model.dim_g) = 0.01*Eigen::MatrixXd::Ones(model.dim_g,model.N);}
        if (model.dim_h) {Y.row(model.dim_g) = 0.001*Eigen::VectorXd::Ones(model.N);}
        if (model.dim_h2) {Y.row(model.dim_c - model.dim_h2) = 0.001*Eigen::VectorXd::Ones(model.N);}
    }

    if (model.S_init.size()) {S = model.S_init;}
    else {
        S = Eigen::MatrixXd::Zero(model.dim_c, model.N);
        if (model.dim_g) {S.topRows(model.dim_g) = 0.01*Eigen::MatrixXd::Ones(model.dim_g,model.N);}
        if (model.dim_h) {S.row(model.dim_g) = 0.001*Eigen::VectorXd::Ones(model.N);}
        if (model.dim_h2) {S.row(model.dim_c - model.dim_h2) = 0.001*Eigen::VectorXd::Ones(model.N);}
    }

    ku.resize(model.dim_u, model.N);
    ky.resize(model.dim_c, model.N);
    ks.resize(model.dim_c, model.N);
    Ku.resize(model.dim_u, model.dim_x * model.N);
    Ky.resize(model.dim_c, model.dim_x * model.N);
    Ks.resize(model.dim_c, model.dim_x * model.N);
}

IPDDP::~IPDDP() {
}

void IPDDP::init(Param param) {
    this->param = param;

    this->initialRoll();
    if (this->param.mu == 0) {this->param.mu = cost / model.N / model.dim_c;} // Auto Select
    this->resetFilter();
    this->resetRegulation();

    for (double i = 1; i < 11; ++i) {
        step_list.push_back(std::pow(2.0, -i));
    }
}

void IPDDP::initialRoll() {
    this->C.resize(this->model.dim_c, this->model.N);
    for (int t = 0; t < this->model.N; ++t) {
        C.col(t) = model.c(X.col(t), U.col(t)).cast<double>();
        X.col(t+1) = model.f(X.col(t), U.col(t)).cast<double>();
    }
    cost = calculateTotalCost(X, U);
}

void IPDDP::resetFilter() {
    double barriercost = 0.0;
    if (model.dim_g) {barriercost += Y.topRows(model.dim_g).array().log().sum();}
    if (model.dim_h) {barriercost += log(Y.row(model.dim_g).array().pow(2.0).sum() - Y.middleRows(model.dim_g+1, model.dim_h-1).array().pow(2.0).sum())/2;}
    if (model.dim_h2) {barriercost += log(Y.row(model.dim_c-model.dim_h2).array().pow(2.0).sum() - Y.bottomRows(model.dim_h2-1).array().pow(2.0).sum())/2;}

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
    for (int t = 0; t < model.N; ++t) {
        cost += model.q(X.col(t), U.col(t));
    }
    cost += model.p(X.col(model.N));
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
    Lx.col(0) = x;
    Lx.row(0) = x.transpose();
    return Lx;
}

void IPDDP::backwardPass() {
    VectorXdual2nd x(model.dim_x);
    VectorXdual2nd u(model.dim_u);
    Eigen::VectorXd y(model.dim_c);
    Eigen::VectorXd s(model.dim_c);
    Eigen::VectorXd c_v(model.dim_c);

    Eigen::VectorXd Vx(model.dim_x);
    Eigen::MatrixXd Vxx(model.dim_x,model.dim_x);

    Eigen::MatrixXd fx(model.dim_x,model.dim_x), fu(model.dim_x,model.dim_u);
    Eigen::MatrixXd Qsx(model.dim_c,model.dim_x), Qsu(model.dim_c,model.dim_u);
    // Eigen::Tensor<double, 3> fxx(model.dim_x,model.dim_x,model.dim_x);
    // Eigen::Tensor<double, 3> fxu(model.dim_x,model.dim_x,model.dim_u);
    // Eigen::Tensor<double, 3> fuu(model.dim_x,model.dim_u,model.dim_u);

    Eigen::VectorXd qx(model.dim_x), qu(model.dim_u);
    Eigen::MatrixXd qxx(model.dim_x,model.dim_x), qxu(model.dim_x,model.dim_u), quu(model.dim_u,model.dim_u);

    Eigen::VectorXd Qx(model.dim_x), Qu(model.dim_u);
    Eigen::MatrixXd Qxx(model.dim_x,model.dim_x), Qxu(model.dim_x,model.dim_u), Quu(model.dim_u,model.dim_u);
    Eigen::MatrixXd Quu_sim(model.dim_u,model.dim_u);

    Eigen::MatrixXd Yinv;
    Eigen::MatrixXd SYinv;

    Eigen::VectorXd rp;
    Eigen::VectorXd rd;
    Eigen::VectorXd r;

    Eigen::LLT<Eigen::MatrixXd> Quu_llt;
    Eigen::MatrixXd R;

    Eigen::VectorXd ku_(model.dim_u);
    Eigen::VectorXd ky_(model.dim_c);
    Eigen::VectorXd ks_(model.dim_c);
    Eigen::MatrixXd Ku_(model.dim_u, model.dim_x);
    Eigen::MatrixXd Ky_(model.dim_c, model.dim_x);
    Eigen::MatrixXd Ks_(model.dim_c, model.dim_x);

    opterror = 0.0;

    // while (true) {
        // dV = Eigen::VectorXd::Zero(2);

        checkRegulate();

        x = X.col(model.N).cast<dual2nd>();
        Vx = gradient(model.p, wrt(x), at(x));
        Vxx = hessian(model.p, wrt(x), at(x));

        // CHECK
        backward_failed = 0;

        for (int t = model.N - 1; t >= 0; --t) {
            int t_dim_x = t * model.dim_x;

            x = X.col(t).cast<dual2nd>();
            u = U.col(t).cast<dual2nd>();

            y = Y.col(t);
            s = S.col(t);
            c_v = C.col(t);

            Eigen::MatrixXd Y_ = Eigen::MatrixXd::Zero(model.dim_c, model.dim_c);
            Eigen::MatrixXd S_ = Eigen::MatrixXd::Zero(model.dim_c, model.dim_c);
            if (model.dim_g) {
                Y_.topLeftCorner(model.dim_g, model.dim_g) = y.topRows(model.dim_g).asDiagonal();
                S_.topLeftCorner(model.dim_g, model.dim_g) = s.topRows(model.dim_g).asDiagonal();
            }
            if (model.dim_h) {
                Y_.block(model.dim_g, model.dim_g, model.dim_h, model.dim_h) = L(y.middleRows(model.dim_g, model.dim_h));
                S_.block(model.dim_g, model.dim_g, model.dim_h, model.dim_h) = L(s.middleRows(model.dim_g, model.dim_h));
            }
            if (model.dim_h2) {
                Y_.bottomRightCorner(model.dim_h2, model.dim_h2) = L(y.bottomRows(model.dim_h2));
                S_.bottomRightCorner(model.dim_h2, model.dim_h2) = L(s.bottomRows(model.dim_h2));
            }
            Eigen::VectorXd e = Eigen::VectorXd::Ones(model.dim_c);
            if (model.dim_h) {
                e.middleRows(model.dim_g+1, model.dim_h-1) = Eigen::VectorXd::Zero(model.dim_h-1);
            }
            if (model.dim_h2) {
                e.bottomRows(model.dim_h2-1) = Eigen::VectorXd::Zero(model.dim_h2-1);
            }
            // std::cout<<"Y\n"<<Y_<<std::endl;
            // std::cout<<"Y_inv\n"<<Y_.inverse()<<std::endl;

            fx = jacobian(model.f, wrt(x), at(x,u));
            fu = jacobian(model.f, wrt(u), at(x,u));

            Qsx = jacobian(model.c, wrt(x), at(x,u));
            Qsu = jacobian(model.c, wrt(u), at(x,u));

            // vectorHessian(fxx, model.f, fs, x, u, "xx");
            // vectorHessian(fxu, model.f, fs, x, u, "xu");
            // vectorHessian(fuu, model.f, fs, x, u, "uu");

            qx = gradient(model.q, wrt(x), at(x,u));
            qu = gradient(model.q, wrt(u), at(x,u));

            Qx = qx + (Qsx.transpose() * s) + (fx.transpose() * Vx);
            Qu = qu + (Qsu.transpose() * s) + (fu.transpose() * Vx);

            qxx = hessian(model.q, wrt(x), at(x,u));
            scalarHessian(qxu, model.q, x, u, "xu");
            quu = hessian(model.q, wrt(u), at(x,u));

            Qxx = qxx + (fx.transpose() * Vxx * fx);
            Qxu = qxu + (fx.transpose() * Vxx * fu);
            Quu = quu + (fu.transpose() * Vxx * fu);
            // std::cout<<"Quu1 = "<<Quu<<std::endl;

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

            Quu += Eigen::MatrixXd::Identity(model.dim_u, model.dim_u) * (std::pow(1.6, regulate) - 1);
            // Quu += Eigen::MatrixXd::Identity(model.dim_u, model.dim_u) * regulate;
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
            Ku.middleCols(t_dim_x, model.dim_x) = Ku_;
            ks.col(t) = ks_;
            Ks.middleCols(t_dim_x, model.dim_x) = Ks_;
            ky.col(t) = ky_;
            Ky.middleCols(t_dim_x, model.dim_x) = Ky_;

            // std::cout<<"ku_: "<<ku_<<std::endl;
            // std::cout<<"Ku_: "<<Ku_<<std::endl;

            std::cout<<"Qu.lpNorm<Eigen::Infinity>(): "<<Qu.lpNorm<Eigen::Infinity>()<<std::endl;
            std::cout<<"rp.lpNorm<Eigen::Infinity>(): "<<rp.lpNorm<Eigen::Infinity>()<<std::endl;
            std::cout<<"rd.lpNorm<Eigen::Infinity>(): "<<rd.lpNorm<Eigen::Infinity>()<<std::endl;
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
    Eigen::MatrixXd X_new(model.dim_x, model.N+1);
    Eigen::MatrixXd U_new(model.dim_u, model.N);
    Eigen::MatrixXd Y_new(model.dim_c, model.N);
    Eigen::MatrixXd S_new(model.dim_c, model.N);
    Eigen::MatrixXd C_new(model.dim_c, model.N);

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
        for (int t = 0; t < model.N; ++t) {
            int t_dim_x = t * model.dim_x;
            Y_new.col(t) = Y.col(t) + (step_size * ky.col(t)) + (Ky.middleCols(t_dim_x, model.dim_x) * (X_new.col(t) - X.col(t)));
            S_new.col(t) = S.col(t) + (step_size * ks.col(t)) + (Ks.middleCols(t_dim_x, model.dim_x) * (X_new.col(t) - X.col(t)));
            if (model.dim_g) {
                if ((Y_new.col(t).topRows(model.dim_g).array() < (1 - tau) * Y.col(t).topRows(model.dim_g).array()).any()) {std::cout<<"1"<<std::endl;forward_failed = true; break;}
                if ((S_new.col(t).topRows(model.dim_g).array() < (1 - tau) * S.col(t).topRows(model.dim_g).array()).any()) {std::cout<<"2"<<std::endl;forward_failed = true; break;}
            }
            if (model.dim_h) {
                if ((Y_new.col(t).row(model.dim_g).array().pow(2.0) - Y_new.col(t).middleRows(model.dim_g+1, model.dim_h-1).array().pow(2.0).sum()
                < (1 - tau) * (Y.col(t).row(model.dim_g).array().pow(2.0) - Y.col(t).middleRows(model.dim_g+1, model.dim_h-1).array().pow(2.0).sum())).any()) {std::cout<<"3"<<std::endl;forward_failed = true; break;}
                if ((S_new.col(t).row(model.dim_g).array().pow(2.0) - S_new.col(t).middleRows(model.dim_g+1, model.dim_h-1).array().pow(2.0).sum()
                < (1 - tau) * (S.col(t).row(model.dim_g).array().pow(2.0) - S.col(t).middleRows(model.dim_g+1, model.dim_h-1).array().pow(2.0).sum())).any()) {std::cout<<"4"<<std::endl;forward_failed = true; break;}
            }
            if (model.dim_h2) {
                if ((Y_new.col(t).row(model.dim_c-model.dim_h2).array().pow(2.0) - Y_new.col(t).bottomRows(model.dim_h2-1).array().pow(2.0).sum()
                < (1 - tau) * (Y.col(t).row(model.dim_c-model.dim_h2).array().pow(2.0) - Y.col(t).bottomRows(model.dim_h2-1).array().pow(2.0).sum())).any()) {std::cout<<"5"<<std::endl;forward_failed = true; break;}
                if ((S_new.col(t).row(model.dim_c-model.dim_h2).array().pow(2.0) - S_new.col(t).bottomRows(model.dim_h2-1).array().pow(2.0).sum()
                < (1 - tau) * (S.col(t).row(model.dim_c-model.dim_h2).array().pow(2.0) - S.col(t).bottomRows(model.dim_h2-1).array().pow(2.0).sum())).any()) {std::cout<<"6"<<std::endl;forward_failed = true; break;}
            }
            U_new.col(t) = U.col(t) + (step_size * ku.col(t)) + (Ku.middleCols(t_dim_x, model.dim_x) * (X_new.col(t) - X.col(t)));
            X_new.col(t+1) = model.f(X_new.col(t), U_new.col(t)).cast<double>();
        }

        if (forward_failed) {continue;}

        cost_new = calculateTotalCost(X_new, U_new);

        barrier_g_new = 0.0;
        barrier_h_new = 0.0;
        if (model.dim_g) {barrier_g_new = Y_new.topRows(model.dim_g).array().log().sum();}
        if (model.dim_h) {barrier_h_new = log(Y_new.row(model.dim_g).array().pow(2.0).sum() - Y_new.middleRows(model.dim_g+1, model.dim_h-1).array().pow(2.0).sum())/2;}
        if (model.dim_h2) {barrier_h_new += log(Y_new.row(model.dim_c-model.dim_h2).array().pow(2.0).sum() - Y_new.bottomRows(model.dim_h2-1).array().pow(2.0).sum())/2;}
        // std::cout<<"barriercost G = "<<barrier_g_new<<std::endl;
        // std::cout<<"barriercost H = "<<barrier_h_new<<std::endl;
        logcost_new = cost_new - param.mu * (barrier_g_new + barrier_h_new);
        for (int t = 0; t < model.N; ++t) {
            C_new.col(t) = model.c(X_new.col(t), U_new.col(t)).cast<double>();
        }
        error = (C + Y).colwise().lpNorm<1>().sum();
        // error_new = std::max(param.tolerance, (C_new + Y_new).lpNorm<1>());
        // if (logcost >= logcost_new || error >= error_new) {break;}
        if (logcost >= logcost_new && error >= error_new) {std::cout<<"10"<<std::endl;break;}
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

Eigen::MatrixXd IPDDP::getResX() {
    return X;
}

Eigen::MatrixXd IPDDP::getResU() {
    return U;
}

std::vector<double> IPDDP::getAllCost() {
    return all_cost;
}