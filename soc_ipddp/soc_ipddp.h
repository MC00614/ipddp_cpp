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
    int dim_x;
    int dim_u;
    Eigen::MatrixXd X;
    Eigen::MatrixXd U;

    // Discrete Time System
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    // Stage Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
    // Terminal Cost Function
    std::function<double(Eigen::VectorXd)> p;
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
        this->backwardPass();
        break;
        this->forwardPass();
        this->regulate();
    }
    
}

void SOC_IPDDP::backwardPass() {
    Eigen::VectorXd Vx = scalarJacobian(p, X.col(N-1));
    Eigen::MatrixXd Vxx = scalarHessian(p, X.col(N-1));

    for (int t = N-1; t >= 0; --t) {
        Eigen::MatrixXd fx = vectorJacobian(f, X.col(t), U.col(t), "x");
        Eigen::MatrixXd fu = vectorJacobian(f, X.col(t), U.col(t), "u");
        Eigen::Tensor<double, 3> fxx = vectorHessian(f, X.col(t), U.col(t), "xx");
        Eigen::Tensor<double, 3> fuu = vectorHessian(f, X.col(t), U.col(t), "uu");
        Eigen::Tensor<double, 3> fxu = vectorHessian(f, X.col(t), U.col(t), "xu");

        Eigen::VectorXd qx = scalarJacobian(q, X.col(t), U.col(t), "x");
        Eigen::VectorXd qu = scalarJacobian(q, X.col(t), U.col(t), "u");
        Eigen::MatrixXd qxx = scalarHessian(q, X.col(t), U.col(t), "xx");
        Eigen::MatrixXd quu = scalarHessian(q, X.col(t), U.col(t), "uu");
        Eigen::MatrixXd qxu = scalarHessian(q, X.col(t), U.col(t), "xu");

        Eigen::VectorXd Qx = qx + fx.transpose()*Vx;
        Eigen::VectorXd Qu = qu + fu.transpose()*Vx;
        Eigen::MatrixXd Qxx = qxx + fx.transpose()*Vxx*fx + tensdot(Vx,fxx);
        Eigen::MatrixXd Quu = quu + fu.transpose()*Vxx*fu + tensdot(Vx,fuu);
        Eigen::MatrixXd Qxu = qxu + fx.transpose()*Vxx*fu + tensdot(Vx,fxu);

        Vx = Qx + Qxu*Quu.inverse()*Qu;
        Vxx = Qxx + Qxu*Quu.inverse()*Qxu.transpose();
        
        // Expose?
        std::cout<<"Vx\n"<<Vx<<std::endl;
        std::cout<<"Vxx\n"<<Vxx<<std::endl;
    }
}

void SOC_IPDDP::forwardPass() {}

void SOC_IPDDP::regulate() {}


Eigen::MatrixXd getX() {

}

Eigen::MatrixXd getU() {
    
}