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
    void setSystemF(Func f);
    template<typename Func>
    void setStageCostQ(Func q);
    template<typename Func>
    void setTerminalCostP(Func p);

    void solve();

    Eigen::MatrixXd getX();
    Eigen::MatrixXd getU();

private:
    int N;
    int max_iter;
    int dim_x;
    int dim_u;
    Eigen::MatrixXd X;
    Eigen::MatrixXd U;

    // Discrete Time System
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    // Stage Cost Function
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> q;
    // Terminal Cost Function
    std::function<Eigen::VectorXd(Eigen::VectorXd)> p;
    void backwardPass();
    void forwardPass();
    void regulate();
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
}


template<typename Func>
void SOC_IPDDP::setSystemF(Func f) {
    this->f = f;
}

template<typename Func>
void SOC_IPDDP::setStageCostQ(Func q) {
    this->q = q;
}

template<typename Func>
void SOC_IPDDP::setTerminalCostP(Func p) {
    this->p = p;
}

void SOC_IPDDP::solve() {
    int iter = 0;

    while (iter++ < this->max_iter) {
        this->backwardPass();
        break;
        this->forwardPass();
        this->regulate();
    }
    
}

void SOC_IPDDP::backwardPass() {    
    Eigen::MatrixXd fx;
    Eigen::MatrixXd fu;
    Eigen::MatrixXd fxx;
    Eigen::MatrixXd fuu;
    Eigen::MatrixXd fxu;

    Eigen::MatrixXd qx;
    Eigen::MatrixXd qu;
    Eigen::MatrixXd qxx;
    Eigen::MatrixXd quu;
    Eigen::MatrixXd qxu;

    Eigen::MatrixXd Qx;
    Eigen::MatrixXd Qu;
    Eigen::MatrixXd Qxx;
    Eigen::MatrixXd Quu;
    Eigen::MatrixXd Qxu;
    Eigen::MatrixXd Vx = calculateJacobian(p, X.col(N-1));
    Eigen::MatrixXd Vxx = calculateHessian(p, X.col(N-1));

    for (int t = N-1; t >= 0; --t) {
        fx = calculateJacobianX(f, X.col(t), U.col(t));
        fu = calculateJacobianU(f, X.col(t), U.col(t));
        fxx = calculateHessianXX(f, X.col(t), U.col(t));
        // qx = calculateJacobianX(q, X.col(t), U.col(t));
        // qu = calculateJacobianU(q, X.col(t), U.col(t));
        // qdd = calculateHessian(q, X.col(t), U.col(t));

        // Qx = qx.transpose() + fx.transpose() * Vx.transpose();
        // Qu = qu.transpose() + fu.transpose() * Vx.transpose();
        // Qxx = qdd
        std::cout << fxx << std::endl;
        // std::cout << Qu << std::endl;
        // std::cout << Vxx.size() << std::endl;
        // std::cout << Vxx << std::endl;
        // std::cout << fd.transpose().rows() << fd.transpose().cols() << std::endl;
        // std::cout << fd.middleCols(0,dim_x).transpose().rows() << fd.middleCols(0,dim_x).transpose().cols() << std::endl;
        // std::cout << Vx.rows() << Vx.cols() << std::endl;
        // std::cout << qd.middleCols(0,dim_x).rows() << qd.middleCols(0,dim_x).cols() << std::endl;
        break;
    }

}

void SOC_IPDDP::forwardPass() {}

void SOC_IPDDP::regulate() {}


Eigen::MatrixXd getX() {

}

Eigen::MatrixXd getU() {
    
}