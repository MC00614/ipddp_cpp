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

    Eigen::MatrixXd getX();
    Eigen::MatrixXd getU();
    std::vector<double> getAllCost();

private:
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

    double total_cost;
    bool is_finished;
    bool in_tolerance;
    Param param;
    void initialRoll();
    void resetFilter();
    double logcost;
    double error;
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
}

SOC_IPDDP::~SOC_IPDDP() {
}

void SOC_IPDDP::init(Param param) {
    this->is_finished = false;
    this->in_tolerance = false;

    this->param = param;

    this->initialRoll();
    if (param.mu==0) {param.mu = total_cost / N / dim_c;} // Auto Select
    this->resetFilter();
    this->resetRegulation();

    this->ku.resize(this->dim_u, this->N);
    this->ky.resize(this->dim_c, this->N);
    this->ks.resize(this->dim_c, this->N);
    this->Ku.resize(this->dim_u, this->dim_x * this->N);
    this->Ky.resize(this->dim_c, this->dim_x * this->N);
    this->Ks.resize(this->dim_c, this->dim_x * this->N);
}

void SOC_IPDDP::initialRoll() {
    this->C.resize(this->dim_c, this->N);
    for (int t = 0; t < this->N; ++t) {
        C.col(t) = c(X.col(t), U.col(t));
        X.col(t+1) = f(X.col(t), U.col(t));
    }
    total_cost = calculateTotalCost(X, U);
}

void SOC_IPDDP::resetFilter() {
    if (param.infeasible) {
        logcost = total_cost - param.mu*Y.array().log().sum();
        error = (C + Y).lpNorm<1>();
        if (error < param.tolerance) {error = 0;}
    }
    else {
        logcost = total_cost - param.mu*(-C).array().log().sum();
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
    double total_cost = 0.0;
    for (int t = 0; t < N; ++t) {
        total_cost += q(X.col(t), U.col(t));
    }
    total_cost += p(X.col(N));
    return total_cost;
}

void SOC_IPDDP::solve() {
    int iter = 0;

    while (iter++ < this->param.max_iter) {
        std::cout<< "\niter : " << iter << std::endl;
        std::cout<< "Backward Pass" << std::endl;
        this->backwardPass();
        std::cout<< "Forward Pass" << std::endl;
        this->forwardPass();
        if (this->in_tolerance) {std::cout<< "In Tolerance" << std::endl; break;}
        if (this->is_finished) {std::cout<< "Finished" << std::endl; break;}
    }
}

void SOC_IPDDP::backwardPass() {
    Eigen::VectorXd dV;
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

    while (true) {
        dV = Eigen::VectorXd::Zero(2);
        c_err = 0;
        mu_err = 0;
        Qu_err = 0;

        checkRegulate();

        Vx = scalarJacobian(p, X.col(N));
        Vxx = scalarHessian(p, X.col(N));

        // CHECK
        backward_failed = 0;

        for (int t = N-1; t >= 0; --t) {
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

            Quu_reg = Quu + (quu*(std::pow(1.6, regulate) - 1));

            if (param.infeasible) {
                Eigen::VectorXd r = S.col(t).array() * Y.col(t).array() - param.mu;
                Eigen::VectorXd r_hat = (S.col(t).array() * (C.col(t)+Y.col(t)).array()).matrix() - r;
                Eigen::VectorXd y_inv = Y.col(t).array().inverse();
                Eigen::MatrixXd diag_sy_inv = (S.col(t).array() * y_inv.array()).matrix().asDiagonal();

                Eigen::LLT<Eigen::MatrixXd> Quu_llt(Quu_reg + (Qsu.transpose() * diag_sy_inv * Qsu));
                if (!Quu.isApprox(Quu.transpose()) || Quu_llt.info() == Eigen::NumericalIssue) {
                    backward_failed = true;
                    break;
                }
                Eigen::MatrixXd R = Quu_llt.matrixU();

                Eigen::MatrixXd term1 = Qu + (Qsu.transpose() * (y_inv.array() * r_hat.array()).matrix());
                Eigen::MatrixXd term2 = Qxu.transpose() + (Qsu.transpose() * diag_sy_inv * Qsx);
                kK = -R.inverse() * (R.transpose().inverse() * (Eigen::MatrixXd(dim_u, 1 + dim_x) << term1, term2).finished());
                ku = kK.leftCols(1);
                Ku = kK.rightCols(dim_x);
                ks = y_inv.array() * (r_hat + S.col(t) * Qsu * ku).array();
                Ks = diag_sy_inv * (Qsx + Qsu * Ku);
                ky = -(C.col(t) + Y.col(t)) - Qsu * ku;
                Ky = -Qsx - Qsu * Ku;

                Quu += Qsu.inverse() * diag_sy_inv * Qsu;
                Qxu += Qsx.inverse() * diag_sy_inv * Qsu;
                Qxx += Qsx.inverse() * diag_sy_inv * Qsx;

                Qu += Qsu.transpose() * (y_inv.array() * r_hat.array()).matrix();
                Qu += Qsx.transpose() * (y_inv.array() * r_hat.array()).matrix();
            }
            else {
                Eigen::VectorXd r = S.col(t) * C.col(t) + param.mu;
                Eigen::VectorXd c_inv = C.col(t).array().inverse();
                Eigen::MatrixXd diag_sc_inv = (S.col(t).array() * c_inv.array()).matrix().asDiagonal();

                Eigen::LLT<Eigen::MatrixXd> Quu_llt(Quu_reg - (Qsu.transpose() * diag_sc_inv * Qsu));
                if (!Quu.isApprox(Quu.transpose()) || Quu_llt.info() == Eigen::NumericalIssue) {
                    backward_failed = true;
                    break;
                }
                Eigen::MatrixXd R = Quu_llt.matrixU();

                Eigen::MatrixXd term1 = Qu + (Qsu.transpose() * (c_inv.array() * r.array()).matrix());
                Eigen::MatrixXd term2 = Qxu.transpose() - (Qsu.transpose() * diag_sc_inv * Qsx);

                kK = -R.inverse() * (R.transpose().inverse() * (Eigen::MatrixXd(dim_u, 1 + dim_x) << term1, term2).finished());
                ku = kK.leftCols(1);
                Ku = kK.rightCols(dim_x);
                ks = -y_inv.array() * (r + S.col(t) * Qsu * ku).array();
                Ks = -diag_sc_inv * (Qsx + Qsu * Ku);
                ky = Eigen::MatrixXd::Zero(dim_c, 1);
                Ky = Eigen::MatrixXd::Zero(dim_c, dim_x);

                Quu -= Qsu.inverse() * diag_sc_inv * Qsu;
                Qxu -= Qsx.inverse() * diag_sc_inv * Qsu;
                Qxx -= Qsx.inverse() * diag_sc_inv * Qsx;

                Qu -= Qsu.transpose() * (c_inv.array() * r.array()).matrix();
                Qu -= Qsx.transpose() * (c_inv.array() * r.array()).matrix();
            }

            dV(0) += ku.transpose() * Qu;
            dV(1) += 0.5 * ku.transpose() * Quu * ku;
            Vx = Qx + (Ku.transpose() * Qu) + (Ku.transpose() * Quu * ku) + (Qxu * ku);
            Vxx = Qxx + (Ku.transpose() * Qxu.transpose()) + (Qxu * Ku) + (Ku.transpose() * Quu * Ku);
        }
        if (!backward_failed) {break;}
    }
}

void SOC_IPDDP::checkRegulate() {
    if (forward_failed || backward_failed) {++regulate;}
    else if (step == 1) {--regulate;}
    else if (5 < step) {++regulate;}

    if (regulate < 0) {regulate = 0;}
    else if (24 < regulate) {regulate = 24;}
}

void SOC_IPDDP::forwardPass() {
    double a = 1.0;
    int max_backtracking_iter = 20;
    int back_tracking_iter = 0;
    double current_cost;
    Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(dim_x, N+1);
    Eigen::MatrixXd U_new = Eigen::MatrixXd::Zero(dim_u, N);
    
    while (back_tracking_iter++ < max_backtracking_iter) {
        X_new.col(0) = X.col(0);
        for (int t = 0; t < N; ++t) {
            U_new.col(t) = U.col(t) + a*k.col(t) + K.middleCols(t * this->dim_x, this->dim_x)*(X_new.col(t) - X.col(t));
            X_new.col(t+1) = f(X_new.col(t), U_new.col(t));
        }
        // X_new.col(N) = X.col(N);
        current_cost = calculateTotalCost(X_new, U_new);
        if (current_cost < total_cost) {
            this->X = X_new;
            this->U = U_new;
            break;
        }
        a = 0.6*a;
    }
    if (back_tracking_iter >= max_backtracking_iter) {this->is_finished = true; return;}
    if (this->total_cost - current_cost < this->param.tolerance) {this->in_tolerance = true;}
    this->all_cost.push_back(current_cost);
    this->total_cost = current_cost;
    std::cout<<"current_cost : "<<current_cost<<std::endl;
}

Eigen::MatrixXd SOC_IPDDP::getX() {
    return X;
}

Eigen::MatrixXd SOC_IPDDP::getU() {
    return U;
}

std::vector<double> SOC_IPDDP::getAllCost() {
    return all_cost;
}