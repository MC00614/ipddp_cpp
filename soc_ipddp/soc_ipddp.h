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

    void init(int N, int max_iter, Eigen::MatrixXd X, Eigen::MatrixXd U);

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
    Eigen::MatrixXd X;
    Eigen::MatrixXd U;
    int dim_x;
    int dim_u;
    int regulate;

    
    Eigen::MatrixXd k;
    Eigen::MatrixXd K;

    // Discrete Time System
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    // Stage Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
    // Terminal Cost Function
    std::function<double(Eigen::VectorXd)> p;
    void backwardPass();
    void forwardPass();
};


SOC_IPDDP::SOC_IPDDP() {
}

SOC_IPDDP::~SOC_IPDDP() {
}

void SOC_IPDDP::init(int N, int max_iter, Eigen::MatrixXd X, Eigen::MatrixXd U) {
    this->N = N;
    this->max_iter = max_iter;
    this->X = X;
    this->U = U;

    this->dim_x = X.rows();
    this->dim_u = U.rows();

    this->k.resize(this->dim_u, this->N);
    this->K.resize(this->dim_u, this->dim_x * this->N);

    this->regulate = 0;
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

void SOC_IPDDP::solve() {
    int iter = 0;

    while (iter++ < this->max_iter) {
        std::cout<< "Backward Pass" << std::endl;
        this->backwardPass();
        std::cout<< "Forward Pass" << std::endl;
        this->forwardPass();
    }
    
}

void SOC_IPDDP::backwardPass() {
    bool backward_failed = true;

    while (backward_failed) {
        Eigen::VectorXd Vx = scalarJacobian(p, X.col(N-1));
        Eigen::MatrixXd Vxx = scalarHessian(p, X.col(N-1));

        for (int t = N-1; t >= 0; --t) {
            Eigen::MatrixXd fx = vectorJacobian(f, X.col(t), U.col(t), "x");
            Eigen::MatrixXd fu = vectorJacobian(f, X.col(t), U.col(t), "u");
            Eigen::Tensor<double, 3> fxx = vectorHessian(f, X.col(t), U.col(t), "xx");
            Eigen::Tensor<double, 3> fxu = vectorHessian(f, X.col(t), U.col(t), "xu");
            Eigen::Tensor<double, 3> fuu = vectorHessian(f, X.col(t), U.col(t), "uu");

            Eigen::VectorXd qx = scalarJacobian(q, X.col(t), U.col(t), "x");
            Eigen::VectorXd qu = scalarJacobian(q, X.col(t), U.col(t), "u");
            Eigen::MatrixXd qxx = scalarHessian(q, X.col(t), U.col(t), "xx");
            Eigen::MatrixXd qxu = scalarHessian(q, X.col(t), U.col(t), "xu");
            Eigen::MatrixXd quu = scalarHessian(q, X.col(t), U.col(t), "uu");

            Eigen::VectorXd Qx = qx + fx.transpose()*Vx;
            Eigen::VectorXd Qu = qu + fu.transpose()*Vx;
            Eigen::MatrixXd Qxx = qxx + fx.transpose()*Vxx*fx + tensdot(Vx,fxx);
            Eigen::MatrixXd Qxu = qxu + fx.transpose()*Vxx*fu + tensdot(Vx,fxu);
            Eigen::MatrixXd Quu = quu + fu.transpose()*Vxx*fu + tensdot(Vx,fuu);

            if (regulate) {Quu += std::pow(1.5, regulate) * Eigen::MatrixXd::Identity(dim_u, dim_u);}
            Eigen::LLT<Eigen::MatrixXd> Quu_llt(Quu);
            if (!Quu.isApprox(Quu.transpose()) || Quu_llt.info() == Eigen::NumericalIssue) {
                regulate += 1;
                break;
            } 
            if (t == 0) {backward_failed = false;}

            this->k.col(t) = -Quu.inverse()*Qu;
            this->K.middleCols(t * this->dim_x, this->dim_x) = -Quu.inverse()*Qxu.transpose();
            // std::cout<<"Quu\n"<<Quu<<std::endl;
            // std::cout<<"Qu\n"<<Qu<<std::endl;
            // std::cout<<"k\n"<<k<<std::endl;
            // std::cout<<"K\n"<<K<<std::endl;

            Vx = Qx - Qxu*Quu.inverse()*Qu;
            Vxx = Qxx - Qxu*Quu.inverse()*Qxu.transpose();
            // std::cout<<"Vx\n"<<Vx<<std::endl;
            // std::cout<<"Vxx\n"<<Vxx<<std::endl;
            // std::cout<<"Quu\n"<<Quu<<std::endl;
            // std::cout<<"Quu.inverse()\n"<<Quu.inverse()<<std::endl;

        }
    }
}

void SOC_IPDDP::forwardPass() {
    // std::cout<<"k\n"<<k<<std::endl;
    // std::cout<<"K\n"<<K<<std::endl;
    Eigen::VectorXd x = X.col(0);
    for (int t = 0; t < N; ++t) {
        U.col(t) = U.col(t) + k.col(t) + K.middleCols(t * this->dim_x, this->dim_x)*(x - X.col(t));
        X.col(t) = x;
        x = f(x, U.col(t));
    }
}

Eigen::MatrixXd SOC_IPDDP::getX() {
    return X;
}

Eigen::MatrixXd SOC_IPDDP::getU() {
    return U;
}