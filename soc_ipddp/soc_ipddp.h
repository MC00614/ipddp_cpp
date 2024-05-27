#include "helper_function.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <functional>
#include <cmath>

#include <iostream>

class SOC_IPDDP {
public:
    SOC_IPDDP();
    ~SOC_IPDDP();

    void init(int N, int max_iter, double cost_tolerance, Eigen::MatrixXd X, Eigen::MatrixXd U);

    template<typename Func>
    void setSystemModel(Func f);
    template<typename Func>
    void setStageCost(Func q);
    template<typename Func>
    void setTerminalCost(Func p);

    void solve();

    Eigen::MatrixXd getX();
    Eigen::MatrixXd getU();

private:
    int N;
    int max_iter;
    double cost_tolerance;
    Eigen::MatrixXd X;
    Eigen::MatrixXd U;
    int dim_x;
    int dim_u;
    int regulate;

    Eigen::MatrixXd k;
    Eigen::MatrixXd K;

    double prev_total_cost;
    bool in_tolerance;

    // Discrete Time System
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    // Stage Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
    // Terminal Cost Function
    std::function<double(Eigen::VectorXd)> p;

    // Algorithm
    void backwardPass();
    void forwardPass();
    double getTotalCost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U);
};


SOC_IPDDP::SOC_IPDDP() {
}

SOC_IPDDP::~SOC_IPDDP() {
    this->regulate = 0;
}

void SOC_IPDDP::init(int N, int max_iter, double cost_tolerance, Eigen::MatrixXd X, Eigen::MatrixXd U) {
    this->N = N;
    this->max_iter = max_iter;
    this->cost_tolerance = cost_tolerance; 
    this->X = X;
    this->U = U;

    this->dim_x = X.rows();
    this->dim_u = U.rows();

    this->k.resize(this->dim_u, this->N);
    this->K.resize(this->dim_u, this->dim_x * this->N);

    this->prev_total_cost = MAXFLOAT;
    this->in_tolerance = false;
}


template<typename Func>
void SOC_IPDDP::setSystemModel(Func f) {
    this->f = f;
}

template<typename Func>
void SOC_IPDDP::setStageCost(Func q) {
    this->q = q;
}

template<typename Func>
void SOC_IPDDP::setTerminalCost(Func p) {
    this->p = p;
}

double SOC_IPDDP::getTotalCost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U) {
    double total_cost = 0.0;
    for (int t = 0; t < N; ++t) {
        total_cost += q(X.col(t), U.col(t));
    }
    total_cost += p(X.col(N));
    return total_cost;
}

void SOC_IPDDP::solve() {
    int iter = 0;

    while (iter++ < this->max_iter) {
        std::cout<< "\niter : " << iter << std::endl;
        std::cout<< "Backward Pass" << std::endl;
        this->backwardPass();
        std::cout<< "Forward Pass" << std::endl;
        this->forwardPass();
        if (this->in_tolerance) {
            std::cout<< "In Tolerance" << std::endl;
            break;
        }
    }
}

void SOC_IPDDP::backwardPass() {
    bool backward_failed = true;
    Eigen::VectorXd Vx;
    Eigen::MatrixXd Vxx;

    Eigen::MatrixXd fx, fu;
    Eigen::Tensor<double, 3> fxx, fxu, fuu;

    Eigen::VectorXd qx, qu;
    Eigen::MatrixXd qxx, qxu, quu;

    Eigen::VectorXd Qx, Qu;
    Eigen::MatrixXd Qxx, Qxu, Quu;
    
    while (backward_failed) {
        Vx = scalarJacobian(p, X.col(N));
        Vxx = scalarHessian(p, X.col(N));

        for (int t = N-1; t >= 0; --t) {
            fx = vectorJacobian(f, X.col(t), U.col(t), "x");
            fu = vectorJacobian(f, X.col(t), U.col(t), "u");
            fxx = vectorHessian(f, X.col(t), U.col(t), "xx");
            fxu = vectorHessian(f, X.col(t), U.col(t), "xu");
            fuu = vectorHessian(f, X.col(t), U.col(t), "uu");

            qx = scalarJacobian(q, X.col(t), U.col(t), "x");
            qu = scalarJacobian(q, X.col(t), U.col(t), "u");
            qxx = scalarHessian(q, X.col(t), U.col(t), "xx");
            qxu = scalarHessian(q, X.col(t), U.col(t), "xu");
            quu = scalarHessian(q, X.col(t), U.col(t), "uu");

            Qx = qx + fx.transpose() * Vx;
            Qu = qu + fu.transpose() * Vx;
            Qxx = qxx + fx.transpose() * Vxx * fx + tensdot(Vx, fxx);
            Qxu = qxu + fx.transpose() * Vxx * fu + tensdot(Vx, fxu);
            Quu = quu + fu.transpose() * Vxx * fu + tensdot(Vx, fuu);

            if (regulate) {Quu += std::pow(1.5, regulate) * Eigen::MatrixXd::Identity(dim_u, dim_u);}
            Eigen::LLT<Eigen::MatrixXd> Quu_llt(Quu);
            if (!Quu.isApprox(Quu.transpose()) || Quu_llt.info() == Eigen::NumericalIssue) {
                regulate += 1;
                break;
            }
            if (t == 0) {backward_failed = false;}

            this->k.col(t) = -Quu.inverse()*Qu;
            this->K.middleCols(t * this->dim_x, this->dim_x) = -Quu.inverse()*Qxu.transpose();

            Vx = Qx - Qxu*Quu.inverse()*Qu;
            Vxx = Qxx - Qxu*Quu.inverse()*Qxu.transpose();
        }
    }
}

void SOC_IPDDP::forwardPass() {
    double a = 1.0;
    int back_tracking_iter = 0;
    double total_cost;
    while (back_tracking_iter++ < 10) {
    // while (back_tracking_iter++ < this->max_backtracking_iter) {
        Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(dim_x, N+1);
        Eigen::MatrixXd U_new = Eigen::MatrixXd::Zero(dim_u, N);
        X_new.col(0) = X.col(0);
        for (int t = 0; t < N; ++t) {
            U_new.col(t) = U.col(t) + a*k.col(t) + K.middleCols(t * this->dim_x, this->dim_x)*(X_new.col(t) - X.col(t));
            X_new.col(t+1) = f(X_new.col(t), U_new.col(t));
        }
        total_cost = getTotalCost(X_new, U_new);
        if (total_cost < prev_total_cost) {
            this->X = X_new;
            this->U = U_new;
            break;
        }
        a = 0.9*a;
    }

    if (this->prev_total_cost - total_cost < this->cost_tolerance) {this->in_tolerance = true;}
    this->prev_total_cost = total_cost;
    std::cout<<"total_cost : "<<total_cost<<std::endl;
}

Eigen::MatrixXd SOC_IPDDP::getX() {
    return X;
}

Eigen::MatrixXd SOC_IPDDP::getU() {
    return U;
}