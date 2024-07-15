#pragma once

#include "param.h"
#include "model_base.h"
#include "helper_function.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <functional>
#include <cmath>

#include <iostream>

class SOC_IPDDP {
public:
    template<typename ModelClass>
    SOC_IPDDP(ModelClass model);
    ~SOC_IPDDP();

    void init(Param param);
    void solve();

    Eigen::MatrixXd getInitX();
    Eigen::MatrixXd getInitU();
    Eigen::MatrixXd getResX();
    Eigen::MatrixXd getResU();
    std::vector<double> getAllCost();

private:
    Eigen::MatrixXd X_init;
    Eigen::MatrixXd U_init;

    int N;
    int dim_x;
    int dim_u;
    int dim_c;
    Eigen::MatrixXd X;
    Eigen::MatrixXd U;
    Eigen::MatrixXd Y;
    Eigen::MatrixXd S;
    Eigen::MatrixXd C;
    // Discrete Time System
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    // Stage Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
    // Terminal Cost Function
    std::function<double(Eigen::VectorXd)> p;
    // Constraint
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> c;

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
    void backwardPass();
    void checkRegulate();
    void forwardPass();
    double calculateTotalCost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U);
};


template<typename ModelClass>
SOC_IPDDP::SOC_IPDDP(ModelClass model) {
    // Check Model
    if (!model.N || !model.dim_x || !model.dim_u) {throw std::invalid_argument("Model Parameter is null.");}
    this->N = model.N;
    this->dim_x = model.dim_x;
    this->dim_u = model.dim_u;
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

SOC_IPDDP::~SOC_IPDDP() {
}

void SOC_IPDDP::init(Param param) {
    this->param = param;

    this->initialRoll();
    if (this->param.mu == 0) {this->param.mu = cost / N / dim_c;} // Auto Select
    this->resetFilter();
    this->resetRegulation();

    for (double i = 1; i < 11; ++i) {
        step_list.push_back(std::pow(2.0, -i));
    }
}

void SOC_IPDDP::initialRoll() {
    this->C.resize(this->dim_c, this->N);
    for (int t = 0; t < this->N; ++t) {
        C.col(t) = c(X.col(t), U.col(t));
        X.col(t+1) = f(X.col(t), U.col(t));
    }
    X_init = X;
    U_init = U;
    cost = calculateTotalCost(X, U);
}

void SOC_IPDDP::resetFilter() {
    if (param.infeasible) {
        logcost = cost - (param.mu*Y.array().log().sum());
        error = (C + Y).lpNorm<1>();
        if (error < param.tolerance) {error = 0;}
    }
    else {
        logcost = cost - (param.mu*(-C).array().log().sum());
        error = 0;
    }
    step = 0;
    forward_failed = false;
}


void SOC_IPDDP::resetRegulation() {
    this->regulate = 0;
    this->backward_failed = false;
}


double SOC_IPDDP::calculateTotalCost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U) {
    double cost = 0.0;
    for (int t = 0; t < N; ++t) {
        cost += q(X.col(t), U.col(t));
    }
    cost += p(X.col(N));
    return cost;
}

void SOC_IPDDP::solve() {
    int iter = 0;

    while (iter++ < this->param.max_iter) {
        std::cout<< "\niter : " << iter << std::endl;
        std::cout<< "Backward Pass" << std::endl;
        this->backwardPass();
        std::cout<< "Forward Pass" << std::endl;
        this->forwardPass();
        std::cout<< "mu : " << param.mu << std::endl;
        std::cout<< "Cost : " << cost << std::endl;
        std::cout<< "Opt Error : " << opterror << std::endl;
        std::cout<< "Regulate : " << regulate << std::endl;
        std::cout<< "Step Size : " << step_list[step] << std::endl;
        all_cost.push_back(cost);

        if (std::max(opterror, param.mu) <= param.tolerance) {
            std::cout << "Optimal Solution" << std::endl;
            break;
        }

        if (opterror <= (0.2 * param.mu)) {
            param.mu = std::max((param.tolerance / 10), std::min(0.2 * param.mu, std::pow(param.mu, 1.2)));
            resetFilter();
            resetRegulation();
        }
    }
}

void SOC_IPDDP::backwardPass() {
    double c_err;
    double mu_err;
    double Qu_err;

    Eigen::VectorXd Vx;
    Eigen::MatrixXd Vxx;

    Eigen::MatrixXd fx, fu;
    Eigen::MatrixXd Qsx, Qsu;
    Eigen::Tensor<double, 3> fxx, fxu, fuu;

    Eigen::VectorXd qx, qu;
    Eigen::MatrixXd qxx, qxu, quu;

    Eigen::VectorXd Qx, Qu;
    Eigen::MatrixXd Qxx, Qxu, Quu;
    Eigen::MatrixXd Quu_reg;

    Eigen::MatrixXd diag_s;
    Eigen::MatrixXd kK;

    Eigen::VectorXd r;

    while (true) {
        // dV = Eigen::VectorXd::Zero(2);
        c_err = 0;
        mu_err = 0;
        Qu_err = 0;

        checkRegulate();

        Vx = scalarJacobian(p, X.col(N));
        Vxx = scalarHessian(p, X.col(N));

        // CHECK
        backward_failed = 0;

        for (int t = N - 1; t >= 0; --t) {
            int t_dim_x = t * dim_x;

            fx = vectorJacobian(f, X.col(t), U.col(t), "x");
            fu = vectorJacobian(f, X.col(t), U.col(t), "u");

            Qsx = vectorJacobian(c, X.col(t), U.col(t), "x");
            Qsu = vectorJacobian(c, X.col(t), U.col(t), "u");

            fxx = vectorHessian(f, X.col(t), U.col(t), "xx");
            fxu = vectorHessian(f, X.col(t), U.col(t), "xu");
            fuu = vectorHessian(f, X.col(t), U.col(t), "uu");

            qx = scalarJacobian(q, X.col(t), U.col(t), "x");
            qu = scalarJacobian(q, X.col(t), U.col(t), "u");

            Qx = qx + (Qsx.transpose() * S.col(t)) + (fx.transpose() * Vx);
            Qu = qu + (Qsu.transpose() * S.col(t)) + (fu.transpose() * Vx);

            qxx = scalarHessian(q, X.col(t), U.col(t), "xx");
            qxu = scalarHessian(q, X.col(t), U.col(t), "xu");
            quu = scalarHessian(q, X.col(t), U.col(t), "uu");

            Qxx = qxx + (fx.transpose() * Vxx * fx) + tensdot(Vx, fxx);
            Qxu = qxu + (fx.transpose() * Vxx * fu) + tensdot(Vx, fxu);
            Quu = quu + (fu.transpose() * Vxx * fu) + tensdot(Vx, fuu);

            diag_s = S.col(t).asDiagonal();

            Quu_reg = Quu + (quu * (std::pow(1.6, regulate) - 1));

            if (param.infeasible) {
                r = S.col(t).array() * Y.col(t).array() - param.mu;
                Eigen::VectorXd r_hat = (S.col(t).array() * (C.col(t) + Y.col(t)).array()).matrix() - r;
                Eigen::VectorXd y_inv = Y.col(t).array().inverse();
                Eigen::MatrixXd diag_sy_inv = (S.col(t).array() * y_inv.array()).matrix().asDiagonal();

                Eigen::LLT<Eigen::MatrixXd> Quu_llt(Quu_reg + (Qsu.transpose() * diag_sy_inv * Qsu));
                if (!Quu.isApprox(Quu.transpose()) || Quu_llt.info() == Eigen::NumericalIssue) {
                    backward_failed = true;
                    break;
                }
                Eigen::MatrixXd R = Quu_llt.matrixU();

                Eigen::MatrixXd row1 = Qu + (Qsu.transpose() * (y_inv.array() * r_hat.array()).matrix());
                Eigen::MatrixXd row2 = Qxu.transpose() + (Qsu.transpose() * diag_sy_inv * Qsx);
                kK = -R.inverse() * (R.transpose().inverse() * (Eigen::MatrixXd(dim_u, 1 + dim_x) << row1, row2).finished());
                ku.col(t) = kK.leftCols(1);
                Ku.middleCols(t_dim_x, dim_x) = kK.rightCols(dim_x);
                ks.col(t) = y_inv.array() * (r_hat + diag_s * Qsu * ku.col(t)).array();
                Ks.middleCols(t_dim_x, dim_x) = diag_sy_inv * (Qsx + Qsu * Ku.middleCols(t_dim_x, dim_x));
                ky.col(t) = -(C.col(t) + Y.col(t)) - Qsu * ku;
                Ky.middleCols(t_dim_x, dim_x) = -Qsx - Qsu * Ku;

                Quu += Qsu.transpose() * diag_sy_inv * Qsu;
                Qxu += Qsx.transpose() * diag_sy_inv * Qsu;
                Qxx += Qsx.transpose() * diag_sy_inv * Qsx;

                Qu += Qsu.transpose() * (y_inv.array() * r_hat.array()).matrix();
                Qx += Qsx.transpose() * (y_inv.array() * r_hat.array()).matrix();
            }
            else {
                r = (diag_s * C.col(t)).array() + param.mu;
                Eigen::VectorXd c_inv = C.col(t).array().inverse();
                Eigen::MatrixXd diag_sc_inv = (S.col(t).array() * c_inv.array()).matrix().asDiagonal();

                Eigen::LLT<Eigen::MatrixXd> Quu_llt(Quu_reg - (Qsu.transpose() * diag_sc_inv * Qsu));
                if (!Quu.isApprox(Quu.transpose()) || Quu_llt.info() == Eigen::NumericalIssue) {
                    backward_failed = true;
                    break;
                }
                Eigen::MatrixXd R = Quu_llt.matrixU();

                Eigen::MatrixXd row1 = Qu - (Qsu.transpose() * (c_inv.array() * r.array()).matrix());
                Eigen::MatrixXd row2 = Qxu.transpose() - (Qsu.transpose() * diag_sc_inv * Qsx);

                kK = -R.inverse() * (R.transpose().inverse() * (Eigen::MatrixXd(dim_u, 1 + dim_x) << row1, row2).finished());
                ku.col(t) = kK.leftCols(1);
                Ku.middleCols(t_dim_x, dim_x) = kK.rightCols(dim_x);
                ks.col(t) = -c_inv.array() * (r + diag_s * Qsu * ku.col(t)).array();
                Ks.middleCols(t_dim_x, dim_x) = -diag_sc_inv * (Qsx + Qsu * Ku.middleCols(t_dim_x, dim_x));
                ky.col(t) = Eigen::MatrixXd::Zero(dim_c, 1);
                Ky.middleCols(t_dim_x, dim_x) = Eigen::MatrixXd::Zero(dim_c, dim_x);

                Quu -= Qsu.transpose() * diag_sc_inv * Qsu;
                Qxu -= Qsx.transpose() * diag_sc_inv * Qsu;
                Qxx -= Qsx.transpose() * diag_sc_inv * Qsx;

                Qu -= Qsu.transpose() * (c_inv.array() * r.array()).matrix();
                Qx -= Qsx.transpose() * (c_inv.array() * r.array()).matrix();
            }

            // dV(0) = dV(0) + (ku.col(t).transpose() * Qu)(0);
            // dV(1) = dV(1) + (0.5 * ku.col(t).transpose() * Quu * ku.col(t))(0);
            Vx = Qx + (Ku.middleCols(t_dim_x, dim_x).transpose() * Qu) + (Ku.middleCols(t_dim_x, dim_x).transpose() * Quu * ku.col(t)) + (Qxu * ku.col(t));
            Vxx = Qxx + (Ku.middleCols(t_dim_x, dim_x).transpose() * Qxu.transpose()) + (Qxu * Ku.middleCols(t_dim_x, dim_x)) + (Ku.middleCols(t_dim_x, dim_x).transpose() * Quu * Ku.middleCols(t_dim_x, dim_x));

            Qu_err = std::max(Qu_err, Qu.lpNorm<Eigen::Infinity>());
            mu_err = std::max(mu_err, r.lpNorm<Eigen::Infinity>());
            if (param.infeasible) {c_err = std::max(c_err, (C.col(t) + Y.col(t)).lpNorm<Eigen::Infinity>());}
        }
        opterror = std::max({Qu_err, mu_err, c_err});
        break;
    }
}

void SOC_IPDDP::checkRegulate() {
    if (forward_failed || backward_failed) {++regulate;}
    else if (step == 0) {--regulate;}
    else if (step <= 3) {regulate = regulate;}
    else {++regulate;}

    if (regulate < 0) {regulate = 0;}
    else if (24 < regulate) {regulate = 24;}
}

void SOC_IPDDP::forwardPass() {
    Eigen::MatrixXd X_new;
    Eigen::MatrixXd U_new;
    Eigen::MatrixXd Y_new;
    Eigen::MatrixXd S_new;
    Eigen::MatrixXd C_new;

    double tau = std::max(0.99, 1.0 - param.mu);
    
    double cost_new = 0.0;
    double logcost_new = 0.0;
    double error_new = 0.0;

    for (step = 0; step < step_list.size(); ++step) {
        X_new = Eigen::MatrixXd::Zero(dim_x, N+1);
        U_new = Eigen::MatrixXd::Zero(dim_u, N);
        Y_new = Eigen::MatrixXd::Zero(dim_c, N);
        S_new = Eigen::MatrixXd::Zero(dim_c, N);
        C_new = Eigen::MatrixXd::Zero(dim_c, N);

        forward_failed = false;
        double step_size = step_list[step];

        X_new.col(0) = X.col(0);
        if (param.infeasible) {
            for (int t = 0; t < N; ++t) {
                int t_dim_x = t * dim_x;
                Y_new.col(t) = Y.col(t) + (step_size * ky.col(t)) + (Ky.middleCols(t_dim_x, dim_x) * (X_new.col(t) - X.col(t)));
                S_new.col(t) = S.col(t) + (step_size * ks.col(t)) + (Ks.middleCols(t_dim_x, dim_x) * (X_new.col(t) - X.col(t)));
                if ((Y_new.col(t).array() < (1 - tau) * Y.col(t).array()).any()) {forward_failed = true; break;}
                if ((S_new.col(t).array() < (1 - tau) * S.col(t).array()).any()) {forward_failed = true; break;}
                U_new.col(t) = U.col(t) + (step_size * ku.col(t)) + (Ku.middleCols(t_dim_x, dim_x) * (X_new.col(t) - X.col(t)));
                X_new.col(t+1) = f(X_new.col(t), U_new.col(t));
            }
        }
        else {
            for (int t = 0; t < N; ++t) {
                int t_dim_x = t * dim_x;
                S_new.col(t) = S.col(t) + (step_size * ks.col(t)) + (Ks.middleCols(t_dim_x, dim_x) * (X_new.col(t) - X.col(t)));
                U_new.col(t) = U.col(t) + (step_size * ku.col(t)) + (Ku.middleCols(t_dim_x, dim_x) * (X_new.col(t) - X.col(t)));
                C_new.col(t) = c(X_new.col(t), U_new.col(t));
                if ((C_new.col(t).array() > (1 - tau) * C.col(t).array()).any()) {forward_failed = true; break;}
                if ((S_new.col(t).array() < (1 - tau) * S.col(t).array()).any()) {forward_failed = true; break;}
                X_new.col(t+1) = f(X_new.col(t), U_new.col(t));
            }
        }

        if (forward_failed) {continue;}

        cost_new = calculateTotalCost(X_new, U_new);

        if (param.infeasible) {
            logcost_new = cost_new - (param.mu * Y_new.array().log().sum());
            // CHECK POSITION
            for (int t = 0; t < N; ++t) {
                C_new.col(t) = c(X_new.col(t), U_new.col(t));
            }
            error_new = std::max(param.tolerance, (C_new + Y_new).lpNorm<1>());
        }
        else {
            logcost_new = cost_new - (param.mu * (-C_new).array().log().sum());
            error_new = 0.0;
        }
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
    }
    else {std::cout<<"Forward Failed"<<std::endl;}
}

Eigen::MatrixXd SOC_IPDDP::getInitX() {
    return X_init;
}

Eigen::MatrixXd SOC_IPDDP::getInitU() {
    return U_init;
}

Eigen::MatrixXd SOC_IPDDP::getResX() {
    return X;
}

Eigen::MatrixXd SOC_IPDDP::getResU() {
    return U;
}

std::vector<double> SOC_IPDDP::getAllCost() {
    return all_cost;
}